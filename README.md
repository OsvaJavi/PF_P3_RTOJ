# Sistema de Predicción de Meteoritos

**Autor:** Oswaldo Javier Rojas Del Toro  
**Materia:** Programación 3  
**Descripción:** Red neuronal para clasificar meteoritos como "Fell" (visto caer) o "Found" (encontrado después)

## Descripción General

Este proyecto implementa un sistema de predicción basado en machine learning supervisado utilizando TensorFlow y Keras. El modelo predice si un meteorito fue observado caer ("Fell") o fue encontrado posteriormente ("Found") basándose en cuatro características:

- **mass (g):** Masa del meteorito en gramos
- **year:** Año de detección o recuperación
- **reclat:** Latitud de recuperación
- **reclong:** Longitud de recuperación

## Características

- **Dataset:** Meteorite_Landings.csv con 45,716 registros de meteoritos
- **Red Neuronal:** Arquitectura de 4 capas densas (64 → 32 → 16 → 1)
- **Normalización:** StandardScaler para estandarizar características
- **Class Weighting:** Pesos balanceados para manejar desbalance de clases
- **Threshold Optimizado:** Búsqueda de umbral óptimo basado en F1-score
- **Métricas:** Accuracy, AUC, Precision, Recall

## Estructura del Proyecto

```
PF_P3_RojasToro/
├── trainer.py          # Entrenamiento del modelo
├── predict.py          # Predicciones con el modelo
├── main.py             # Interfaz principal con menú
├── requirements.txt    # Dependencias
├── README.md           # Este archivo
├── .gitignore          # Archivos a ignorar en Git
└── Meteorite_Landings.csv  # Dataset (en tu máquina local)
```

## Instalación

### Requisitos previos
- Python 3.11+ (recomendado para TensorFlow)
- pip (gestor de paquetes)

### Pasos de instalación

1. **Crea un entorno virtual:**
   ```bash
   python -m venv venv
   ```

2. **Activa el entorno virtual:**
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Asegúrate de que `Meteorite_Landings.csv` está en el mismo directorio que los scripts**

## Uso

### Ejecución principal
```bash
python main.py
```

Esto abrirá un menú interactivo con las siguientes opciones:

1. **Entrenar nuevo modelo**
   - Carga el dataset
   - Entrena la red neuronal
   - Busca el threshold óptimo
   - Guarda el modelo, scaler y threshold

2. **Predicción desde sensores (simulación)**
   - Simula lecturas de 4 sensores
   - Realiza predicciones en tiempo real
   - Muestra confianza y probabilidades

3. **Predicción desde CSV**
   - Permite predecir sobre una fila específica del dataset
   - Compara la predicción con el valor real

4. **Salir**

### Ejecución individual de módulos

**Entrenar modelo:**
```bash
python -m trainer
```

**Hacer una predicción (uso avanzado):**
```python
import predict

resultado = predict.predecir([500.5, 2000, -33.5, 20.4])
print(resultado)
```

## Archivos generados durante el entrenamiento

Después de entrenar el modelo, se generan tres archivos:

- `meteorite_model.keras` - Modelo entrenado de TensorFlow
- `meteorite_scaler.pkl` - Scaler para normalización de datos
- `meteorite_threshold.pkl` - Threshold óptimo para clasificación

## Pipeline de datos

```
Dataset CSV
    ↓
Carga y limpieza (dropna)
    ↓
Normalización (StandardScaler)
    ↓
Split train/val (80/20)
    ↓
Entrenamiento de red neuronal
    ↓
Búsqueda de threshold óptimo
    ↓
Evaluación con métricas
    ↓
Guardado de modelo, scaler y threshold
```

## Notas técnicas

### Balanceo de clases
El dataset tiene un desbalance entre "Fell" (~22%) y "Found" (~78%). Se usa `compute_class_weight()` para aplicar pesos inversamente proporcionales durante el entrenamiento.

### Threshold optimizado
En lugar de usar 0.5 como umbral, se busca el valor que maximiza el F1-score para la clase "Fell", mejorando la capacidad de detección.

### Early Stopping
Se implementa `EarlyStopping` monitoreando el AUC de validación con paciencia de 5 épocas para evitar overfitting.

## Licencia

Este proyecto es para propósitos educativos.

## Autor

Oswaldo Javier Rojas Del Toro - Programación 3 - Universidad de Guadalajara (CUGDL)
