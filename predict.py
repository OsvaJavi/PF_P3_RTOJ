import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


MODEL_PATH = "meteorite_model.keras"
SCALER_PATH = "meteorite_scaler.pkl"
THRESHOLD_PATH = "meteorite_threshold.pkl"
DATASET_PATH = "Meteorite_Landings.csv"


def cargar_modelo_y_metadatos(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    threshold_path=THRESHOLD_PATH,
):
    """Carga el modelo, scaler y threshold guardados."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(threshold_path):
        return None

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    threshold = joblib.load(threshold_path)

    return model, scaler, threshold


def predecir(
    features,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    threshold_path=THRESHOLD_PATH,
):
    """
    Realiza una predicción sobre si un meteorito fue visto caer (Fell) o encontrado (Found).
    
    Args:
        features: List o array con [mass_g, year, reclat, reclong]
        model_path: Ruta al modelo guardado
        scaler_path: Ruta al scaler guardado
        threshold_path: Ruta al threshold guardado
    
    Returns:
        Dict con etiqueta, confianza y probabilidades
    """
    cargado = cargar_modelo_y_metadatos(model_path, scaler_path, threshold_path)
    
    if cargado is None:
        raise FileNotFoundError(
            "No se encontró modelo o metadatos. Ejecuta primero el entrenamiento."
        )

    model, scaler, threshold = cargado

    # Normalizar los datos
    x = np.asarray(features, dtype=np.float32).reshape(1, 4)
    x_norm = scaler.transform(x)

    # Hacer predicción
    prob = model.predict(x_norm, verbose=0)[0][0]
    clase = 1 if prob > threshold else 0
    etiqueta = "Fell" if clase == 1 else "Found"

    return {
        "etiqueta": etiqueta,
        "clase_numerica": int(clase),
        "confianza": float(prob),
        "threshold_usado": float(threshold),
        "probabilidades": {
            "Found": float(1 - prob),
            "Fell": float(prob),
        },
    }


def predecir_desde_fila_csv(
    row_index=0,
    dataset_path=DATASET_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    threshold_path=THRESHOLD_PATH,
):
    """
    Lee una fila del CSV y hace una predicción.
    
    Args:
        row_index: Índice de la fila a predecir
        dataset_path: Ruta al dataset
    
    Returns:
        Dict con etiqueta, confianza y probabilidades
    """
    data = pd.read_csv(dataset_path)
    data = data.dropna(subset=['mass (g)', 'year', 'reclat', 'reclong'])
    
    if row_index >= len(data):
        raise IndexError(f"Índice {row_index} fuera de rango. Dataset tiene {len(data)} filas.")
    
    features = data.iloc[row_index, :][['mass (g)', 'year', 'reclat', 'reclong']].astype(np.float32).to_numpy()
    resultado = predecir(features, model_path, scaler_path, threshold_path)
    
    # Agregar información de la fila
    nombre = data.iloc[row_index]['name']
    resultado['nombre'] = nombre
    resultado['actual'] = data.iloc[row_index]['fall']
    
    return resultado
