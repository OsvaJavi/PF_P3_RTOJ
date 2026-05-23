import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib


MODEL_PATH = "meteorite_model.keras"
SCALER_PATH = "meteorite_scaler.pkl"
THRESHOLD_PATH = "meteorite_threshold.pkl"
DATASET_PATH = "Meteorite_Landings.csv"


def cargar_datos(dataset_path=DATASET_PATH):
    """Carga y prepara los datos del dataset de meteoritos."""
    datos = pd.read_csv(dataset_path)
    datos = datos.dropna(subset=['mass (g)', 'year', 'reclat', 'reclong', 'fall'])

    X = datos[['mass (g)', 'year', 'reclat', 'reclong']].astype(np.float32).to_numpy()
    y = (datos['fall'] == 'Fell').astype(np.float32).to_numpy()

    print("\nEstadísticas básicas del dataset:")
    for i, col in enumerate(['mass (g)', 'year', 'reclat', 'reclong']):
        print(f"  {col:15s} media: {np.mean(X[:, i]):.4f}")
    
    fell = int(y.sum())
    found = int((1 - y).sum())
    total = fell + found
    print(f"\nDistribución target:")
    print(f"  Fell  (1): {fell:>6,}  ({fell/total*100:.1f}%)")
    print(f"  Found (0): {found:>6,}  ({found/total*100:.1f}%)\n")

    return X, y


def construir_modelo():
    """Construye la arquitectura de la red neuronal."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return model


def entrenar_modelo(X, y, epochs=50, batch_size=64):
    """Entrena el modelo con los datos proporcionados."""
    # Normalizar datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Balanced class weights
    pesos = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight = {0: pesos[0], 1: pesos[1]}
    print(f"Class weight aplicado → Found: {pesos[0]:.3f} | Fell: {pesos[1]:.3f}\n")

    # Construir y entrenar
    model = construir_modelo()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    )

    print("Iniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stop],
    )

    # Buscar threshold óptimo basado en F1
    print("\nBuscando threshold óptimo (F1 de Fell)...")
    y_pred_prob = model.predict(X_val, verbose=0).flatten()

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [
        f1_score(y_val, (y_pred_prob > t).astype(int), pos_label=1, zero_division=0)
        for t in thresholds
    ]

    best_threshold = float(thresholds[np.argmax(f1_scores)])
    best_f1 = max(f1_scores)
    print(f"Threshold óptimo: {best_threshold:.2f} → F1 Fell: {best_f1:.4f}")
    print(f"(threshold por defecto era 0.50)\n")

    # Evaluar con el threshold óptimo
    print("Evaluación en set de validación:")
    y_pred = (y_pred_prob > best_threshold).astype(int)
    print(classification_report(y_val, y_pred, target_names=['Found (0)', 'Fell (1)']))
    
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print("           Found  Fell")
    cm = confusion_matrix(y_val, y_pred)
    print(f"Found (real): {cm[0,0]:>5}  {cm[0,1]:>5}")
    print(f"Fell  (real): {cm[1,0]:>5}  {cm[1,1]:>5}\n")

    return model, scaler, best_threshold, history


def guardar_modelo(
    model,
    scaler,
    best_threshold,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    threshold_path=THRESHOLD_PATH,
):
    """Guarda el modelo, scaler y threshold."""
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(best_threshold, threshold_path)
    
    print(f"✓ Modelo guardado       → {model_path}")
    print(f"✓ Scaler guardado       → {scaler_path}")
    print(f"✓ Threshold guardado    → {threshold_path} (valor: {best_threshold:.2f})")


def entrenar_y_guardar(
    dataset_path=DATASET_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    threshold_path=THRESHOLD_PATH,
):
    """Pipeline completo: carga, entrena y guarda."""
    print("="*80)
    print("ENTRENAMIENTO DE MODELO - PREDICCIÓN DE METEORITOS")
    print("="*80)
    
    X, y = cargar_datos(dataset_path)
    model, scaler, best_threshold, _ = entrenar_modelo(X, y)
    guardar_modelo(model, scaler, best_threshold, model_path, scaler_path, threshold_path)
    
    print("\n" + "="*80)
    print("¡Entrenamiento completado!")
    print("="*80)


if __name__ == "__main__":
    entrenar_y_guardar()
