# Trabajo Práctico 2 - Inferencia de Clima en Australia

Este proyecto implementa un pipeline de inferencia para predecir si lloverá mañana en Australia, utilizando modelos de Machine Learning y Deep Learning.

El sistema está contenerizado con Docker para facilitar su despliegue y ejecución.

## Descripción del Proyecto

El script principal `inference.py` realiza las siguientes tareas:
1.  **Carga de Artefactos**: Modelos entrenados (Red Neuronal Keras), escaladores, encoders y configuraciones de imputación.
2.  **Preprocesamiento**: Limpieza de datos, imputación de valores faltantes (usando técnicas cruzadas y por cluster), ingeniería de características y transformación de datos.
3.  **Inferencia**: Predicción de probabilidad de lluvia usando el modelo de Red Neuronal.
4.  **Resultados**: Muestra los resultados en consola y genera un archivo `predictions.csv`.

## Estructura del Directorio Docker

- `inference.py`: Script de inferencia.
- `Dockerfile`: Definición de la imagen Docker.
- `requirements.txt`: Dependencias de Python.
- `*.joblib` / `*.keras`: Modelos y artefactos serializados.
- `input.csv`: Archivo de entrada por defecto (ejemplo).

## Instrucciones de Uso con Docker

### Prerrequisitos

- Tener [Docker](https://www.docker.com/get-started) instalado y ejecutándose en tu máquina.

- Clonar el repositorio:
```bash
git clone https://github.com/TobiasPrieto/AA1-TUIA-Prieto-Esteva.git
```
### 1. Construir la Imagen

Navega a la carpeta `docker` en tu terminal y ejecuta el siguiente comando para construir la imagen:

BASH
```bash
cd docker
docker build -t prediccion-clima .
```

Esto creará una imagen llamada `prediccion-clima`.

### 2. Ejecutar el Contenedor

#### Opción A: Ejecutar con el archivo de prueba interno

Para probar que todo funciona correctamente usando el archivo `input.csv` que ya está dentro de la imagen:

POWERSHELL
```PowerShell
docker run --rm -v $(pwd):/app prediccion-clima
```


#### Opcion B: Ejecutar con nuevos datos

Si se quiere predecir con datos alternativos a los proporcionados por defecto, se debe sobreescribir el archivo `input.csv` con los datos deseados, respetando las columnas ya existentes y luego ejecutar el siguiente comando:

POWERSHELL
```PowerShell
docker run --rm -v "$(pwd):/app" prediccion-clima
```

o 

CMD
```cmd
docker run --rm -v "%cd%:/app" prediccion-clima
```



## Formato de Entrada

El archivo CSV de entrada debe contener las columnas esperadas por el modelo, las cuales son:
- Date
- Location
- MinTemp
- MaxTemp
- Rainfall
- Evaporation
- Sunshine
- WindGustDir
- WindGustSpeed
- WindDir9am
- WindDir3pm
- WindSpeed9am
- WindSpeed3pm
- Humidity9am
- Humidity3pm
- Pressure9am
- Pressure3pm
- Cloud9am
- Cloud3pm
- Temp9am
- Temp3pm
- RainToday
- RainTomorrow
