import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class DataLoader:
    def __init__(self, csv_path="data/processed/train.csv", processed_dir="data/processed"):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en {csv_path}")

        self.processed_dir = Path(processed_dir)

        # cosas que se llenan después
        self.preproc = None
        self.input_dim = None
        self._X_columns = None

        # crear carpeta si no existe
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True)

        print("[INFO] DataLoader inicializado")


    def _validate_and_map_labels(self, y: np.ndarray) -> np.ndarray:
        # revisar labels
        uniq = set(np.unique(y))

        # por si vienen como 1 / 2
        if uniq.issubset({1, 2}):
            y = (y - 1).astype(int)

        # check final
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("label must be 0/1 (or 1/2)")

        return y.astype(int)


    def build_preprocessor(self, X_df: pd.DataFrame):
        # columnas numericas
        numeric_cols = X_df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        # columnas categoricas
        categorical_cols = X_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # pipeline numerico
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers = []
        transformers.append(("num", num_pipe, numeric_cols))

        # pipeline categorico solo si hay columnas
        if categorical_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
            ])
            transformers.append(("cat", cat_pipe, categorical_cols))

        preproc = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0
        )

        return preproc, numeric_cols, categorical_cols


    def load_and_preprocess(
        self,
        target_col: str = "label",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        print("[INFO] Cargando dataset:", self.csv_path)

        df = pd.read_csv(self.csv_path)

        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in {self.csv_path}")

        # labels
        y_raw = df[target_col].astype(int).values
        y = self._validate_and_map_labels(y_raw)

        # features
        X_df = df.drop(columns=[target_col]).copy()
        self._X_columns = X_df.columns.tolist()

        # construir preprocesador
        preproc, num_cols, cat_cols = self.build_preprocessor(X_df)

        # fit + transform
        X_all = preproc.fit_transform(X_df)

        self.preproc = preproc
        self.input_dim = X_all.shape[1]

        print("[INFO] Dimensión de entrada:", self.input_dim)

        # split train / val
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return (X_train, y_train), (X_val, y_val)


    def save_preproc(self, path: str):
        if self.preproc is None:
            raise RuntimeError("No preproc to save. Run load_and_preprocess() first.")

        try:
            joblib.dump(self.preproc, path)
            print("[INFO] Preprocesador guardado en:", path)
        except Exception as e:
            print("[WARN] No se pudo guardar el preprocesador:", e)


    def load_preproc(self, path: str):
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"No se encontró preproc en {path}")

        self.preproc = joblib.load(path)
        print("[INFO] Preprocesador cargado desde:", path)


    def transform_df(self, df: pd.DataFrame):
        if self.preproc is None:
            raise RuntimeError("Preprocessor not loaded. Run load_and_preprocess() or load_preproc().")

        return self.preproc.transform(df)
