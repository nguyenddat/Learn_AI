from typing import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class TfLinearRegression:
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: int,
            batch_size: int,
            epochs: int,
            lr: float
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self._build_model()

    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray
    ) -> None:
        history = self.model.fit(
            x = X,
            y = y,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_split = 0.2
        )
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mse = history.history['mse']
        val_mse = history.history['val_mse']

        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.plot(epochs, mse, 'g--', label='Training MSE')
        plt.plot(epochs, val_mse, 'y--', label='Validation MSE')

        plt.fill_between(epochs, loss, val_loss, color='gray', alpha=0.2)
        plt.title('Training and Validation Loss & MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.show()

    def predict(
            self, 
            X: np.ndarray
        ):
        return self.model.predict(X)

    def fit_predict(self, X, y):
        history = self.model.fit(
            x = X,
            y = y,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_split = 0.2
        )
        return self.model.predict(X)

    def _build_model(self):
        input = tf.keras.Input(shape = self.input_shape)

        output = tf.keras.layers.Dense(units = self.output_shape, activation = "linear", use_bias = True)(input)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(
                learning_rate = self.lr,
                momentum = 0.0
            ),
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.MeanSquaredError(name = "mse")]
        )
        self.model.summary()
        return self