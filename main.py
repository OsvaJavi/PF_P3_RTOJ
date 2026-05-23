import random
import time
import os

import predict
import trainer

MODEL_PATH = "meteorite_model.keras"
SCALER_PATH = "meteorite_scaler.pkl"
THRESHOLD_PATH = "meteorite_threshold.pkl"


def simular_mediciones_sensores():
    """
    Simula la lectura de 4 sensores correspondientes a las características de un meteorito:
    - mass (g): masa en gramos (escala logarítmica simulada)
    - year: año de detección
    - reclat: latitud de recuperación
    - reclong: longitud de recuperación
    """
    # Simulación realista basada en rangos del dataset
    mass_g = random.lognormvariate(2, 2)  # Distribución log-normal para masa
    year = random.uniform(1860, 2020)
    reclat = random.uniform(-90, 90)
    reclong = random.uniform(-180, 180)

    return mass_g, year, reclat, reclong


def interfaz_prediccion_desde_sensores():
    """
    Interfaz principal: simula la lectura de sensores y realiza predicciones en tiempo real.
    """
    print("\n" + "="*80)
    print("PREDICCIÓN DE METEORITOS DESDE SENSORES (SIMULACIÓN)")
    print("="*80)

    prediccion_num = 1
    while True:
        try:
            mass_g, year, reclat, reclong = simular_mediciones_sensores()
            resultado = predict.predecir([mass_g, year, reclat, reclong])

            print(f"\n[Predicción #{prediccion_num}]")
            print("-" * 80)
            print("Mediciones de sensores:")
            print(f"  Mass (g):      {mass_g:.2f}")
            print(f"  Year:          {year:.0f}")
            print(f"  Reclat:        {reclat:.4f}")
            print(f"  Reclong:       {reclong:.4f}")

            print("\nResultado de predicción:")
            print(f"  Etiqueta:      {resultado['etiqueta']}")
            print(f"  Confianza:     {resultado['confianza']:.4f}")
            print(f"  Threshold:     {resultado['threshold_usado']:.2f}")
            print(f"  Probabilidades:")
            print(f"    - Found:     {resultado['probabilidades']['Found']:.4f}")
            print(f"    - Fell:      {resultado['probabilidades']['Fell']:.4f}")

            prediccion_num += 1
            time.sleep(2)  # Esperar 2 segundos entre predicciones

        except KeyboardInterrupt:
            print("\n\n¡Simulación detenida por el usuario!")
            break
        except Exception as e:
            print(f"\nError durante la predicción: {e}")
            break


def menu_principal():
    """Menú interactivo principal."""
    while True:
        print("\n" + "="*80)
        print("SISTEMA DE PREDICCIÓN DE METEORITOS")
        print("="*80)
        print("\n1. Entrenar nuevo modelo")
        print("2. Predicción desde sensores (simulación)")
        print("3. Predicción desde CSV")
        print("4. Salir")
        print("\nSelecciona una opción: ", end="")

        try:
            opcion = input().strip()

            if opcion == "1":
                print("\nEntrenando modelo...")
                trainer.entrenar_y_guardar()

            elif opcion == "2":
                # Verificar si existe modelo
                if not os.path.exists(MODEL_PATH):
                    print("\n⚠️  No existe modelo entrenado.")
                    print("Por favor, entrena el modelo primero (opción 1).")
                else:
                    interfaz_prediccion_desde_sensores()

            elif opcion == "3":
                if not os.path.exists(MODEL_PATH):
                    print("\n⚠️  No existe modelo entrenado.")
                    print("Por favor, entrena el modelo primero (opción 1).")
                else:
                    print("\nIngresa el índice de fila a predecir (0-indexed): ", end="")
                    try:
                        row_idx = int(input().strip())
                        resultado = predict.predecir_desde_fila_csv(row_idx)

                        print("\n" + "-"*80)
                        print(f"Nombre del meteorito:  {resultado['nombre']}")
                        print(f"Valor actual (real):   {resultado['actual']}")
                        print(f"\nPredicción:")
                        print(f"  Etiqueta:            {resultado['etiqueta']}")
                        print(f"  Confianza:           {resultado['confianza']:.4f}")
                        print(f"  Threshold usado:     {resultado['threshold_usado']:.2f}")
                        print(f"  Probabilidades:")
                        print(f"    - Found:           {resultado['probabilidades']['Found']:.4f}")
                        print(f"    - Fell:            {resultado['probabilidades']['Fell']:.4f}")
                        print("-"*80)
                    except ValueError:
                        print("Error: Por favor ingresa un número válido.")
                    except IndexError as e:
                        print(f"Error: {e}")

            elif opcion == "4":
                print("\n¡Hasta luego!")
                break

            else:
                print("\n⚠️  Opción no válida. Por favor selecciona 1, 2, 3 o 4.")

        except KeyboardInterrupt:
            print("\n\n¡Programa interrumpido!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Función principal."""
    # Verificar si existe modelo
    cargado = predict.cargar_modelo_y_metadatos(MODEL_PATH, SCALER_PATH, THRESHOLD_PATH)

    if cargado is None:
        print("\n" + "="*80)
        print("BIENVENIDO AL SISTEMA DE PREDICCIÓN DE METEORITOS")
        print("="*80)
        print("\n⚠️  No existe modelo entrenado.")
        print("Se procederá con el entrenamiento...\n")
        trainer.entrenar_y_guardar()
    else:
        print("\n" + "="*80)
        print("BIENVENIDO AL SISTEMA DE PREDICCIÓN DE METEORITOS")
        print("="*80)
        print("\n✓ Modelo y metadatos cargados exitosamente.")

    # Mostrar menú
    menu_principal()


if __name__ == "__main__":
    main()
