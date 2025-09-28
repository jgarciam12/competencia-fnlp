# Importamos las librerías necesarias
import pandas as pd                           # para manejar los datos en DataFrames
from sklearn.model_selection import train_test_split  # para dividir en train y validación
from sklearn.feature_extraction.text import TfidfVectorizer  # para convertir texto en números
from sklearn.linear_model import LogisticRegression          # modelo de clasificación
from sklearn.metrics import classification_report, accuracy_score  # métricas de evaluación
import joblib                              # para guardar el modelo entrenado
import argparse                            # para recibir parámetros desde la terminal
from pathlib import Path                   # para manejar rutas de carpetas

def main(args):
    # Leer el dataset de entrenamiento
    df = pd.read_csv(args.data)
    # X = columna con los textos
    X = df["text"]
    # y = columna con la etiqueta (target)
    y = df[args.target]

    # Dividir en train (80%) y validación (20%), manteniendo balance de clases (stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Transformar texto a vectores usando TF-IDF
    # - max_features=10000: máximo 10k palabras/características
    # - ngram_range=(1,2): usa unigramas (palabras sueltas) y bigramas (pares de palabras)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)  # ajusta y transforma train
    X_val_vec = vectorizer.transform(X_val)          # transforma validación con el mismo vocabulario

    # Crear el modelo: Regresión Logística (buena base para texto)
    model = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto")
    model.fit(X_train_vec, y_train)  # entrenar el modelo

    # Evaluar el modelo en validación
    preds = model.predict(X_val_vec)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    # Guardar modelo y vectorizador juntos en un archivo .joblib
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)  # crea carpeta si no existe
    joblib.dump({"model": model, "vectorizer": vectorizer}, args.out)
    print(f"Modelo guardado en {args.out}")

# Si ejecutamos desde la terminal (python src/train.py ...)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv", help="Ruta al CSV de entrenamiento")
    parser.add_argument("--target", default="decade", help="Columna objetivo")
    parser.add_argument("--out", default="models/model.joblib", help="Ruta para guardar el modelo")
    args = parser.parse_args()

    main(args)  # ejecutar entrenamiento
