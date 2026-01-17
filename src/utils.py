import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class Utils:
    @staticmethod
    def seed_everything(seed=42):
        # a veces cambio la semilla para probar, pero por defecto dejo 42
        if seed is None:
            seed = 42

        random.seed(seed)
        np.random.seed(seed)

        try:
            tf.random.set_seed(seed)
        except Exception as e:
            print("[WARN] No se pudo fijar la semilla de tensorflow", e)

        os.environ["PYTHONHASHSEED"] = str(seed)

        print("[INFO] Semilla global fijada en:", seed)

    # calcula las metricas de evaluacion del modelo
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # por si vienen como listas
        if isinstance(y_true, list):
            y_true = np.array(y_true)

        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        # métricas básicas para binario
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        # calculo f1 aparte por claridad
        f1 = f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)

        print("\n=== MÉTRICAS DEL MODELO ===")

        if precision >= 0:
            print(f"Precisión: {precision:.4f}")
        else:
            print("Precisión rara:", precision)

        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("Matriz de confusión:")
        print(cm)

        metrics = {}
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1
        metrics["confusion_matrix"] = cm

        return metrics

    # guarda las métricas en un archivo de texto para revisar después
    @staticmethod
    def save_metrics(metrics_dict, save_path="experiments/metrics.txt"):
        folder = os.path.dirname(save_path)

        if folder != "":
            if not os.path.exists(folder):
                os.makedirs(folder)

        with open(save_path, "w") as f:
            for key in metrics_dict:
                value = metrics_dict[key]
                f.write(str(key))
                f.write(": ")
                f.write(str(value))
                f.write("\n")

        print("[INFO] Métricas guardadas en:", save_path)

    # resumen rápido del entrenamiento
    @staticmethod
    def print_summary(train_loss, val_loss, train_acc, val_acc):
        print("\n=== RESUMEN DE ENTRENAMIENTO ===")

        # a veces las listas vienen vacías si algo falló
        if len(train_loss) > 0:
            last_train_loss = train_loss[-1]
        else:
            last_train_loss = None

        if len(val_loss) > 0:
            last_val_loss = val_loss[-1]
        else:
            last_val_loss = None

        print("Pérdida entrenamiento:", round(last_train_loss, 4) if last_train_loss is not None else "N/A")
        print("Pérdida validación:", round(last_val_loss, 4) if last_val_loss is not None else "N/A")

        if train_acc is not None and len(train_acc) > 0:
            print(f"Precisión entrenamiento: {train_acc[-1]:.4f}")

        if val_acc is not None and len(val_acc) > 0:
            print(f"Precisión validación: {val_acc[-1]:.4f}")
