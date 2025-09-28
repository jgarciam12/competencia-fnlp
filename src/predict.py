# Importamos librerías necesarias
import argparse         # para leer parámetros desde terminal
import pandas as pd     # para cargar CSVs
import joblib           # para cargar el modelo entrenado
from pathlib import Path # para manejar rutas y crear carpetas

def main(args):
    # Cargar el modelo entrenado (y el vectorizador) desde .joblib
    model_data = joblib.load(args.model)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]

    # Leer el archivo de evaluación (eval.csv)
    df = pd.read_csv(args.input)
    # La columna de entrada es "text"
    X = df["text"]

    # Transformar textos a vectores usando el mismo vectorizador del entrenamiento
    X_vec = vectorizer.transform(X)
    # Obtener predicciones con el modelo
    preds = model.predict(X_vec)

    # Crear DataFrame con id y predicciones (columna decade)
    submission = pd.DataFrame({
        "id": df["id"],        # copiamos el id original de eval
        "decade": preds        # añadimos la predicción
    })

    # Guardar el archivo submission.csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)  # crear carpeta si no existe
    submission.to_csv(args.out, index=False)
    print(f"Archivo de submission guardado en {args.out}")

# Si ejecutamos desde terminal (python src/predict.py ...)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib", help="Ruta al modelo entrenado")
    parser.add_argument("--input", default="data/eval.csv", help="CSV con datos para predecir")
    parser.add_argument("--out", default="submission.csv", help="Archivo de salida con predicciones")
    args = parser.parse_args()

    main(args)  # ejecutar predicción