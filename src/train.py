import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model import MLPhishModel
from .data_loader import DataLoader
from .utils import Utils


class Trainer:
    def __init__(self, raw_data_path="data/raw/dataset.csv", processed_dir="data/processed", seed=42):
        # paths principales
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir

        # por si seed viene raro
        if seed is None:
            seed = 42
        self.seed = seed

        # crear carpetas necesarias
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if not os.path.exists("experiments"):
            os.makedirs("experiments")

        # fijamos semillas
        Utils.seed_everything(self.seed)

        # loader de datos
        self.loader = DataLoader(raw_data_path)

        # variables que se llenan después
        self.scaler = None
        self.model = None
        self.history = None

        print("[INFO] Trainer inicializado")

    # aca se carga y preprocesa los datos
    def prepare_data(self):
        print("[INFO] Cargando dataset...")
        df = self.loader.load_csv()

        # por si acaso viene vacío
        if df is None or len(df) == 0:
            raise ValueError("Dataset vacío o no cargado correctamente")

        # separar features y label
        X = df.drop(columns=["label"]).values
        y = df["label"].astype(int).values

        # split train / val
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed
        )

        # escalado
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # guardar scaler
        scaler_path = os.path.join(self.processed_dir, "scaler.joblib")
        try:
            joblib.dump(self.scaler, scaler_path)
            print("[INFO] Escalador guardado en:", scaler_path)
        except Exception as e:
            print("[WARN] No se pudo guardar el escalador", e)

        # guardar arrays procesados
        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train_scaled)
        np.save(os.path.join(self.processed_dir, "X_val.npy"), X_val_scaled)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, "y_val.npy"), y_val)

        print("[INFO] Datos procesados guardados en", self.processed_dir)

        return X_train_scaled, X_val_scaled, y_train, y_val

    # entrena el modelo MLPhish
    def train_model(self, X_train, X_val, y_train, y_val):
        print("[INFO] Creando modelo...")

        # input dim
        input_dim = X_train.shape[1]

        # crear modelo
        self.model = MLPhishModel(
            input_dim=input_dim,
            learning_rate=0.001
        )

        # construir y compilar
        self.model.build_model()
        self.model.compile_model()

        print("[INFO] Entrenando modelo...")

        # entrenar
        self.history = self.model.train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=25,
            batch_size=32,
            save_path="experiments/best_model.keras"
        )

        if self.history is not None:
            print("[INFO] Entrenamiento completado exitosamente")
        else:
            print("[WARN] El entrenamiento no devolvió historial")

    # ejecuta todo el pipeline
    def run(self):
        print("[INFO] Iniciando entrenamiento de MLPhish...")

        data = self.prepare_data()

        # unpack manual para debug
        X_train = data[0]
        X_val = data[1]
        y_train = data[2]
        y_val = data[3]

        self.train_model(X_train, X_val, y_train, y_val)

        print("[INFO] Pipeline finalizado correctamente")


# bloque principal
if __name__ == "__main__":
    print("[INFO] Ejecutando Trainer desde shell")

    trainer = Trainer(
        raw_data_path="data/raw/dataset.csv",
        processed_dir="data/processed",
        seed=42
    )

    trainer.run()