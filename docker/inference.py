import pandas as pd
import numpy as np
import joblib
import sys
import os
# Suprimimos logs de TensorFlow para mantener la consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from datetime import datetime

def cargar_artefactos():
    print("Cargando artefactos exactos y Red Neuronal...")
    
    # 1. Cargar configuraciones estadísticas (Lookups, Ratios, Medianas)
    try:
        config = joblib.load('artifacts_exact.joblib')
    except FileNotFoundError:
        print("Error: No se encontró 'artifacts_exact.joblib'.")
        sys.exit(1)

    # 2. Cargar transformadores de Scikit-Learn
    kmeans = joblib.load('kmeans.joblib')
    encoder = joblib.load('encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # 3. Cargar Modelo de Red Neuronal (Keras)
    try:
        model = tf.keras.models.load_model('model_nn.keras')
    except Exception as e:
        print(f"Error cargando la red neuronal: {e}")
        sys.exit(1)
    
    # 4. Cargar coordenadas auxiliares (opcional pero recomendado)
    try:
        coords_df = pd.read_csv('coordenadas_aus.csv')
    except:
        coords_df = None
        
    return config, kmeans, encoder, scaler, model, coords_df

def preprocesar_datos(df, artifacts, kmeans, encoder, scaler, coords_df=None):
    # ---------------------------------------------------------
    # 0. PREPARACIÓN INICIAL
    # ---------------------------------------------------------
    # Merge con coordenadas si existen
    if coords_df is not None and 'Latitud' not in df.columns and 'Location' in df.columns:
        df = df.merge(coords_df, on="Location", how="left")
        
    # Fechas
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    # Clustering
    if 'Latitud' not in df.columns: df['Latitud'] = np.nan
    if 'Longitud' not in df.columns: df['Longitud'] = np.nan
    
    # Rellenar coordenadas con 0 para que KMeans no falle (fallback)
    coords_pred = df[['Latitud', 'Longitud']].fillna(0)
    
    try:
        df['RegionCluster'] = kmeans.predict(coords_pred)
    except:
        df['RegionCluster'] = 0
        
    # Corrección específica para NorfolkIsland (como en el TP)
    if 'NorfolkIsland' in df['Location'].values:
         df.loc[df['Location'] == 'NorfolkIsland', 'RegionCluster'] = 5

    # ---------------------------------------------------------
    # 1. IMPUTACIÓN EXACTA (Usando Artifacts)
    # ---------------------------------------------------------
    print("Aplicando imputación exacta...")
    
    # A. Imputación Cruzada por Ratios (Cross-Variable)
    ratios = artifacts['ratios']
    
    # Limpiar Clouds (9 octas es inválido -> NaN)
    if 'Cloud9am' in df.columns: df.loc[df['Cloud9am'] == 9, 'Cloud9am'] = np.nan
    if 'Cloud3pm' in df.columns: df.loc[df['Cloud3pm'] == 9, 'Cloud3pm'] = np.nan

    cross_imputes = [
        ('Humidity9am', 'Humidity3pm', 'humidity_9am_to_3pm', 'humidity_3pm_to_9am'),
        ('MinTemp', 'MaxTemp', 'min_to_max', 'max_to_min'),
        ('Temp9am', 'Temp3pm', 'temp_9am_to_3pm', 'temp_3pm_to_9am'),
        ('Pressure9am', 'Pressure3pm', 'pressure_9am_to_3pm', 'pressure_3pm_to_9am'),
        ('Cloud9am', 'Cloud3pm', 'cloud_9am_to_3pm', 'cloud_3pm_to_9am')
    ]

    for v1, v2, r1, r2 in cross_imputes:
        if v1 in df.columns and v2 in df.columns:
            # Si falta v1, usar v2 * ratio
            mask_1 = df[v1].isna() & df[v2].notna()
            df.loc[mask_1, v1] = df.loc[mask_1, v2] * ratios[r1]
            # Si falta v2, usar v1 * ratio
            mask_2 = df[v2].isna() & df[v1].notna()
            df.loc[mask_2, v2] = df.loc[mask_2, v1] * ratios[r2]

    # B. Imputación por Tablas de Búsqueda (Lookups)
    
    # 1. Lookup Location/Month
    if 'loc_month' in artifacts['lookups']:
        lookup_lm = artifacts['lookups']['loc_month']
        df = df.merge(lookup_lm, on=['Location', 'Month'], how='left', suffixes=('', '_imp'))
        for col in lookup_lm.columns:
            if col not in ['Location', 'Month'] and col in df.columns:
                df[col] = df[col].fillna(df[f'{col}_imp'])
                # Fallback: Mediana Global
                if col in artifacts['global_medians']:
                    df[col] = df[col].fillna(artifacts['global_medians'][col])
        df.drop(columns=[c for c in df.columns if '_imp' in c], inplace=True)

    # 2. Lookup Cluster/Month
    if 'cluster_month' in artifacts['lookups']:
        lookup_cm = artifacts['lookups']['cluster_month']
        df = df.merge(lookup_cm, on=['RegionCluster', 'Month'], how='left', suffixes=('', '_imp'))
        for col in lookup_cm.columns:
            if col not in ['RegionCluster', 'Month'] and col in df.columns:
                df[col] = df[col].fillna(df[f'{col}_imp'])
                if col in artifacts['global_medians']:
                    df[col] = df[col].fillna(artifacts['global_medians'][col])
        df.drop(columns=[c for c in df.columns if '_imp' in c], inplace=True)

    # C. Imputación Especial: WindGustSpeed
    if 'WindGustSpeed' in df.columns:
        max_wind = df[['WindSpeed9am', 'WindSpeed3pm']].max(axis=1)
        df['WindGustSpeed'] = df['WindGustSpeed'].fillna(max_wind)
    
    # D. Imputación Categórica (Moda por Cluster)
    if 'cluster_modes' in artifacts['lookups']:
        lookup_modes = artifacts['lookups']['cluster_modes']
        df = df.merge(lookup_modes, on=['RegionCluster'], how='left', suffixes=('', '_mode'))
        cols_cat = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        for col in cols_cat:
            if col in df.columns:
                df[col] = df[col].fillna(df[f'{col}_mode'])
        df.drop(columns=[c for c in df.columns if '_mode' in c], inplace=True)

    # Lógica RainToday (Si llovió > 0mm es Yes)
    if 'Rainfall' in df.columns:
        df.loc[df['Rainfall'] > 0, 'RainToday'] = 'Yes'
        df.loc[df['Rainfall'] == 0, 'RainToday'] = 'No'
    
    # Mapeo RainToday a 0/1
    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0)

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ---------------------------------------------------------
    wind_map = artifacts['wind_dir_map']
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if col in df.columns:
            # Mapear y rellenar desconocidos con 0
            df[col] = df[col].map(wind_map).fillna(0) 
            df[f'{col}_sin'] = np.sin(np.deg2rad(df[col]))
            df[f'{col}_cos'] = np.cos(np.deg2rad(df[col]))

    # Diferencias
    pairs = [
        ('Temp9am', 'Temp3pm', 'temp_diff'),
        ('Humidity9am', 'Humidity3pm', 'humidity_diff'),
        ('WindSpeed9am', 'WindSpeed3pm', 'wind_diff'),
        ('MinTemp', 'MaxTemp', 'min_max_diff'),
        ('Cloud9am', 'Cloud3pm', 'cloud_diff'),
        ('Pressure9am', 'Pressure3pm', 'pressure_diff')
    ]
    for c1, c2, diff_col in pairs:
        if c1 in df.columns and c2 in df.columns:
            df[diff_col] = abs(df[c1] - df[c2])
        else:
            df[diff_col] = 0

    # ---------------------------------------------------------
    # 3. ENCODING & SCALING
    # ---------------------------------------------------------
    # One Hot Encoding (Month, RegionCluster)
    cols_cat_ohe = ['Month', 'RegionCluster']
    encoded_array = encoder.transform(df[cols_cat_ohe])
    encoded_cols = encoder.get_feature_names_out(cols_cat_ohe)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=cols_cat_ohe), df_encoded], axis=1)

    # Scaling
    features_escalar = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm','temp_diff',
        'humidity_diff', 'wind_diff', 'min_max_diff', 'cloud_diff', 'pressure_diff']
    
    # Asegurar que Cloud sea numérico
    df['Cloud9am'] = df['Cloud9am'].astype(float)
    df['Cloud3pm'] = df['Cloud3pm'].astype(float)
    
    # Rellenar cualquier NaN remanente en features numéricas con 0 antes de escalar
    for f in features_escalar:
        if f not in df.columns: df[f] = 0
        df[f] = df[f].fillna(0)

    df[features_escalar] = scaler.transform(df[features_escalar])
    
    # Eliminar columnas que no se usan en el modelo
    # NOTA: Mantenemos RainToday, features escaladas, sen/cos y OHE
    cols_drop = ['Date','Location','WindGustDir','WindDir9am','WindDir3pm','Latitud','Longitud']
    cols_drop = [c for c in cols_drop if c in df.columns]
    
    X = df.drop(columns=cols_drop)
    
    # Convertir a float32 para TensorFlow
    return X.astype('float32')

if __name__ == "__main__":
    # 1. Cargar recursos
    config, kmeans, encoder, scaler, model, coords_df = cargar_artefactos()

    # 2. Cargar datos de entrada
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'input.csv'
    
    try:
        df_input = pd.read_csv(input_file)
        print(f"Leyendo datos desde: {input_file}")
    except:
        # Generar dato dummy si no hay CSV (Para prueba rápida)
        print(f"No se encontró {input_file}. ")
       
    
    try:
        # 3. Procesar
        X = preprocesar_datos(df_input, config, kmeans, encoder, scaler, coords_df)
        
        # 4. Predecir con Red Neuronal
        # predict devuelve probabilidades (array de arrays)
        predictions_proba = model.predict(X, verbose=0)
        
        print(f"\n{'='*40}")
        print("RESULTADOS DE INFERENCIA (RED NEURONAL)")
        print(f"{'='*40}")
        
        for i, prob_arr in enumerate(predictions_proba):
            prob_lluvia = prob_arr[0] # Probabilidad de clase 1
            pred_clase = 1 if prob_lluvia > 0.5 else 0
            
            res = "LLOVERÁ" if pred_clase == 1 else "NO LLOVERÁ"
            loc = df_input.iloc[i].get('Location', 'Desconocida')
            date = df_input.iloc[i].get('Date', 'Hoy')
            
            print(f"Registro {i+1}: {loc} ({date})")
            print(f"  Pronóstico: {res}")
            print(f"  Probabilidad: {prob_lluvia:.2%}")
            print("-" * 30)
            
        print(f"{'='*40}\n")
        
    except Exception as e:
        print(f"Error crítico en inferencia: {e}")
        import traceback
        traceback.print_exc()