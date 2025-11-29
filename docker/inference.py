import pandas as pd
import numpy as np
import joblib
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from datetime import datetime

def cargar_artefactos():
    print("Cargando artefactos exactos y Red Neuronal...")
    
    try:
        config = joblib.load('artifacts_exact.joblib')
        kmeans = joblib.load('kmeans.joblib')
        encoder = joblib.load('encoder.joblib')
        scaler = joblib.load('scaler.joblib')
        model = tf.keras.models.load_model('model_nn.keras')
    except Exception as e:
        print(f"Error cargando artefactos: {e}")
        sys.exit(1)
    
    try:
        coords_df = pd.read_csv('coordenadas_aus.csv')
    except:
        coords_df = None
        
    return config, kmeans, encoder, scaler, model, coords_df

def preprocesar_datos(df, artifacts, kmeans, encoder, scaler, coords_df=None):

    if 'RainTomorrow' in df.columns:
        df.drop(columns=['RainTomorrow'], inplace=True)

    # Coordenadas y Fechas
    if coords_df is not None:
        df = df.merge(coords_df, on="Location", how="left")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    # Clustering
    coords_pred = df[['Latitud', 'Longitud']].fillna(0)
    df['RegionCluster'] = kmeans.predict(coords_pred)
    
    # Corrección NorfolkIsland
    mask_norfolk = df['Location'] == 'NorfolkIsland'
    if mask_norfolk.any():
         df.loc[mask_norfolk, 'RegionCluster'] = 5

    #Imputación exacta
    ratios = artifacts['ratios']
    
    # Limpiamos Clouds (9 octas -> NaN)
    df.loc[df['Cloud9am'] == 9, 'Cloud9am'] = np.nan
    df.loc[df['Cloud3pm'] == 9, 'Cloud3pm'] = np.nan

    # Imputación Cruzada (Ratios)
    cross_imputes = [
        ('Humidity9am', 'Humidity3pm', 'humidity_9am_to_3pm', 'humidity_3pm_to_9am'),
        ('MinTemp', 'MaxTemp', 'min_to_max', 'max_to_min'),
        ('Temp9am', 'Temp3pm', 'temp_9am_to_3pm', 'temp_3pm_to_9am'),
        ('Pressure9am', 'Pressure3pm', 'pressure_9am_to_3pm', 'pressure_3pm_to_9am'),
        ('Cloud9am', 'Cloud3pm', 'cloud_9am_to_3pm', 'cloud_3pm_to_9am')
    ]

    for v1, v2, r1, r2 in cross_imputes:
        # Imputamos v1 usando v2
        mask_1 = df[v1].isna() & df[v2].notna()
        df.loc[mask_1, v1] = df.loc[mask_1, v2] * ratios[r1]
        # Imputamos v2 usando v1
        mask_2 = df[v2].isna() & df[v1].notna()
        df.loc[mask_2, v2] = df.loc[mask_2, v1] * ratios[r2]

    # Imputación por Tablas (Lookups)

    # Location/Month
    lookup_lm = artifacts['lookups']['loc_month']
    df = df.merge(lookup_lm, on=['Location', 'Month'], how='left', suffixes=('', '_imp'))
    for col in lookup_lm.columns:
        if col not in ['Location', 'Month']:
            df[col] = df[col].fillna(df[f'{col}_imp'])
            # Fallback global
            if col in artifacts['global_medians']:
                df[col] = df[col].fillna(artifacts['global_medians'][col])
    df.drop(columns=[c for c in df.columns if '_imp' in c], inplace=True)

    # Cluster/Month
    lookup_cm = artifacts['lookups']['cluster_month']
    df = df.merge(lookup_cm, on=['RegionCluster', 'Month'], how='left', suffixes=('', '_imp'))
    for col in lookup_cm.columns:
        if col not in ['RegionCluster', 'Month']:
            df[col] = df[col].fillna(df[f'{col}_imp'])
            if col in artifacts['global_medians']:
                df[col] = df[col].fillna(artifacts['global_medians'][col])
    df.drop(columns=[c for c in df.columns if '_imp' in c], inplace=True)

    # WindGustSpeed
    max_wind = df[['WindSpeed9am', 'WindSpeed3pm']].max(axis=1)
    df['WindGustSpeed'] = df['WindGustSpeed'].fillna(max_wind)
    
    # Categóricas (Moda por Cluster)
    lookup_modes = artifacts['lookups']['cluster_modes']
    df = df.merge(lookup_modes, on=['RegionCluster'], how='left', suffixes=('', '_mode'))
    cols_cat = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in cols_cat:
        df[col] = df[col].fillna(df[f'{col}_mode'])
    df.drop(columns=[c for c in df.columns if '_mode' in c], inplace=True)

    # Corrección RainToday basada en Rainfall (prioridad sobre el valor original)
    mask_rain_pos = df['Rainfall'] > 0
    mask_rain_zero = df['Rainfall'] == 0
    df.loc[mask_rain_pos, 'RainToday'] = 'Yes'
    df.loc[mask_rain_zero, 'RainToday'] = 'No'
    
    # Mapeo a binario
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1}).fillna(0)

    # Feature Engineering
    wind_map = artifacts['wind_dir_map']
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        df[col] = df[col].map(wind_map).fillna(0) 
        df[f'{col}_sin'] = np.sin(np.deg2rad(df[col]))
        df[f'{col}_cos'] = np.cos(np.deg2rad(df[col]))

    # Diferencias absolutas
    pairs = [
        ('Temp9am', 'Temp3pm', 'temp_diff'),
        ('Humidity9am', 'Humidity3pm', 'humidity_diff'),
        ('WindSpeed9am', 'WindSpeed3pm', 'wind_diff'),
        ('MinTemp', 'MaxTemp', 'min_max_diff'),
        ('Cloud9am', 'Cloud3pm', 'cloud_diff'),
        ('Pressure9am', 'Pressure3pm', 'pressure_diff')
    ]
    for c1, c2, diff_col in pairs:
        df[diff_col] = abs(df[c1] - df[c2])

    # Encoding & Scaling

    cols_cat_ohe = ['Month', 'RegionCluster']
    encoded_array = encoder.transform(df[cols_cat_ohe])
    encoded_cols = encoder.get_feature_names_out(cols_cat_ohe)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=cols_cat_ohe), df_encoded], axis=1)

    features_escalar = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm','temp_diff',
        'humidity_diff', 'wind_diff', 'min_max_diff', 'cloud_diff', 'pressure_diff']
    
    # Aseguramos tipos float
    df[features_escalar] = df[features_escalar].astype(float)
    
    # Imputamos cualquier NaN restante (por seguridad antes de escalar) con 0
    df[features_escalar] = df[features_escalar].fillna(0)

    df[features_escalar] = scaler.transform(df[features_escalar])
    
    # Eliminamos columnas crudas
    cols_drop = ['Date','Location','WindGustDir','WindDir9am','WindDir3pm','Latitud','Longitud']
    X = df.drop(columns=cols_drop)
    
    # Alinear columnas con el modelo
    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]
    
    return X.astype('float32')

if __name__ == "__main__":
    # Cargamos recursos
    config, kmeans, encoder, scaler, model, coords_df = cargar_artefactos()

    # Cargamos datos
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'input.csv'
    
    try:
        df_input = pd.read_csv(input_file)
        print(f"Leyendo datos desde: {input_file}")
    except Exception as e:
        print(f"Error leyendo {input_file}: {e}")
        sys.exit(1)
    
    try:
        # Procesamos datos
        X = preprocesar_datos(df_input.copy(), config, kmeans, encoder, scaler, coords_df)
        
        # Predecimos
        predictions_proba = model.predict(X, verbose=0)
        
        # Generamos resultados
        results = []
        
        print(f"\n{'='*40}")
        print("RESULTADOS DE INFERENCIA")
        print(f"{'='*40}")
        
        for i, prob_arr in enumerate(predictions_proba):
            prob_lluvia = float(prob_arr[0])
            prob_no_lluvia = 1.0 - prob_lluvia
            pred_etiqueta = "Yes" if prob_lluvia > 0.5 else "No"
            
            loc = df_input.iloc[i]['Location']
            date_val = df_input.iloc[i]['Date']
            
            print(f"Registro {i+1}: {loc} ({date_val}) -> {pred_etiqueta} ({prob_lluvia:.2%})")
            
            results.append({
                'Location': loc,
                'Date': date_val,
                'Prediccion': pred_etiqueta,
                'Probabilidad_Lluvia': prob_lluvia,
                'Probabilidad_No_Lluvia': prob_no_lluvia
            })
            
        print(f"{'='*40}\n")
        
        # Guardamos CSV
        output_csv = 'predictions.csv'
        df_resultados = pd.DataFrame(results)
        df_resultados.to_csv(output_csv, index=False)
        print(f"Archivo generado: {output_csv}")

    except Exception as e:
        print(f"Error en inferencia: {e}")
        import traceback
        traceback.print_exc()