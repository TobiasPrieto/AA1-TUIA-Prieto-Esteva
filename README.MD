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

### 1. Construir la Imagen

Navega a la carpeta `docker` en tu terminal y ejecuta el siguiente comando para construir la imagen:

```bash
cd docker
docker build -t prediccion-clima .
```

Esto creará una imagen llamada `prediccion-clima`.

### 2. Ejecutar el Contenedor

#### Opción A: Ejecutar con el archivo de prueba interno

Para probar que todo funciona correctamente usando el archivo `input.csv` que ya está dentro de la imagen:

```bash
docker run --rm prediccion-clima
```

#### Opción B: Ejecutar con tu propio archivo de datos

Si tienes un archivo de datos nuevo (por ejemplo `mis_datos.csv`) y quieres obtener predicciones sobre él, debes montar el volumen para compartir el archivo con el contenedor.

Supongamos que tienes `mis_datos.csv` en tu carpeta actual:

**En Linux/Mac/PowerShell:**
```bash
docker run --rm -v ${PWD}:/app/data prediccion-clima python inference.py /app/data/mis_datos.csv
```

**En Windows Command Prompt (cmd):**
```cmd
docker run --rm -v %cd%:/app/data prediccion-clima python inference.py /app/data/mis_datos.csv
```

> **Nota**: Los resultados se mostrarán en la consola. Si deseas guardar el archivo `predictions.csv` generado en tu máquina local, asegúrate de montar el directorio donde quieres que se guarde.

Ejemplo para guardar el output en tu carpeta actual:

```bash
docker run --rm -v ${PWD}:/app prediccion-clima
```
*Al montar el directorio actual en `/app`, el script leerá el `input.csv` local (si existe) y escribirá `predictions.csv` en tu directorio actual.*

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
