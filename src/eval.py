import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model


class Evaluator:
    def __init__(self, model_path="experiments/best_model.keras", preproc_path="data/processed/scaler.joblib"):
        # rutas
        self.model_path = Path(model_path)
        self.preproc_path = Path(preproc_path)

        self.model = None
        self.preproc = None

        # checks básicos
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo entrenado en {self.model_path}")

        if not self.preproc_path.exists():
            raise FileNotFoundError(f"No se encontró el preprocesador en {self.preproc_path}")

        # cargar modelo
        try:
            self.model = load_model(self.model_path)
            print("[INFO] Modelo cargado desde:", self.model_path)
        except Exception as e:
            print("[ERROR] No se pudo cargar el modelo")
            raise e

        # cargar preprocesador
        try:
            self.preproc = joblib.load(self.preproc_path)
            print("[INFO] Preprocesador cargado desde:", self.preproc_path)
        except Exception as e:
            print("[ERROR] No se pudo cargar el preprocesador")
            raise e


    def load_data(self, csv_path: str, label_col: str = "label"):
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {csv_path}")

        df = pd.read_csv(csv_path)

        if label_col not in df.columns:
            raise KeyError(f"El CSV debe contener una columna '{label_col}'")

        # separar X e y
        y_true = df[label_col].astype(int).values
        X_raw = df.drop(columns=[label_col])

        return X_raw, y_true


    def preprocess_data(self, X_raw: pd.DataFrame):
        # transformar datos usando el scaler guardado
        X_transformed = self.preproc.transform(X_raw)
        return X_transformed


    def predict(self, X_transformed: np.ndarray):
        # predicciones crudas
        preds = self.model.predict(X_transformed)

        # binarizar manualmente
        preds_binary = (preds > 0.5).astype(int)

        return preds, preds_binary


    def evaluate(
        self,
        csv_path: str,
        label_col: str = "label",
        save_path: str = "experiments/eval_results.csv"
    ):
        print("[INFO] Iniciando evaluación...")

        # cargar datos
        X_raw, y_true = self.load_data(csv_path, label_col)

        # preprocesar
        X_transformed = self.preprocess_data(X_raw)

        # predecir
        preds, preds_binary = self.predict(X_transformed)

        # flatten por si viene con forma rara
        y_pred = preds_binary.flatten()

        # métricas
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # roc usa las probabilidades
        roc = roc_auc_score(y_true, preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary"
        )

        print("\n=== RESULTADOS DE EVALUACIÓN ===")
        print(pd.DataFrame(report).transpose())
        print("\nMatriz de confusión:")
        print(cm)
        print(f"ROC AUC: {roc:.4f}")
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

        # guardar resultados simples
        results_df = pd.DataFrame({
            "metric": ["roc_auc", "precision", "recall", "f1"],
            "value": [roc, precision, recall, f1]
        })

        try:
            results_df.to_csv(save_path, index=False)
            print("[INFO] Resultados guardados en:", save_path)
        except Exception as e:
            print("[WARN] No se pudieron guardar los resultados:", e)

        return report, cm, roc
