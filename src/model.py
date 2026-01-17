import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


class MLPhishModel:
    def __init__(self, input_dim, learning_rate=0.001):
        self.input_dim = input_dim

        # por si el learning rate viene raro
        if learning_rate <= 0:
            learning_rate = 0.001
        self.learning_rate = learning_rate

        # aca se almacenara el modelo una vez construido
        self.model = None

        print("[INFO] MLPhishModel inicializado")

    # metodo para construir el modelo
    def build_model(self):
        # por si input_dim no esta bien definido
        if self.input_dim is None or self.input_dim <= 0:
            raise ValueError("input_dim no válido")

        # definimos una red neuronal secuencial sencilla
        self.model = models.Sequential()

        # capa de entrada + primera capa densa
        self.model.add(
            layers.Dense(
                64,
                activation='relu',
                input_shape=(self.input_dim,),
                kernel_regularizer=regularizers.l2(0.001)
            )
        )

        # dropout para evitar sobreajuste
        self.model.add(layers.Dropout(0.3))

        # segunda capa densa
        self.model.add(
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            )
        )

        self.model.add(layers.Dropout(0.3))

        # capa de salida (binaria)
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print("[INFO] Modelo MLPhish construido exitosamente")

        return self.model

    def compile_model(self):
        if self.model is None:
            raise ValueError("Primero debes construir el modelo con build_model().")

        # optimizador
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # compilamos el modelo
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("[INFO] Modelo MLPhish compilado y listo para entrenar")

        return self.model

    # metodo para entrenar el modelo
    def train_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        batch_size=32,
        save_path="experiments/best_model.keras"
    ):
        if self.model is None:
            raise ValueError("Debes compilar el modelo antes de entrenarlo.")

        print("[INFO] Iniciando entrenamiento...")

        # entrenamiento del modelo
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # guardar modelo entrenado
        try:
            self.model.save(save_path)
            print("[INFO] Modelo entrenado y guardado en", save_path)
        except Exception as e:
            print("[WARN] No se pudo guardar el modelo:", e)

        return history

    # carga un modelo ya entrenado
    def load_trained_model(self, path="experiments/best_model.keras"):
        if not path:
            raise ValueError("Ruta del modelo no válida")

        self.model = tf.keras.models.load_model(path)

        print("[INFO] Modelo cargado desde", path)

        return self.model

    # metodo para hacer predicciones
    def predict(self, X):
        if self.model is None:
            raise ValueError("Debes cargar o entrenar un modelo antes de predecir.")

        preds = self.model.predict(X)

        # conversion manual a 0 y 1
        preds_bin = (preds > 0.5).astype(int)

        return preds_bin

    # evalua el modelo con datos nuevos
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Debes cargar o entrenar un modelo antes de evaluar.")

        results = self.model.evaluate(X_test, y_test, verbose=0)

        loss = results[0]
        acc = results[1]

        print(f"[INFO] Pérdida (loss): {loss:.4f} | Precisión (accuracy): {acc:.4f}")

        return loss, acc
