from typing import *

import tensorflow as tf
import matplotlib.pyplot as plt

from Regression_Model import RegressionModel

class ANN(RegressionModel):
    def __init__(self, 
        input_shape: Tuple[int, int], 
        output_shape: int, 
        learning_rate: float = 0.01,
        epochs: int = 30,
        batch_size: int = 16
):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self._build_model()
    
    def _build_model(self):
        input = tf.keras.layers.Input(shape = self.input_shape)

        x = tf.keras.layers.Dense(units = 8, activation = "relu")(input)
        x = tf.keras.layers.Dense(units = 8, activation = "relu")(x)

        output = tf.keras.layers.Dense(units = self.output_shape)(x)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = self.learning_rate),
            loss = "mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        self.model.summary()
        return self
    

    def fit(self, X, y, plot = True):
        history = self.model.fit(
            x = X,
            y = y,
            validation_split = 0.2,
            epochs = self.epochs,
            batch_size = self.batch_size,
            verbose = 1
        )

        if plot:
            plt.figure(figsize = (20, 10))
            plt.plot(history.history["val_root_mean_squared_error"], label = "val_rmse", marker = "x", color = "red", alpha = 0.4, linestyle = "-")
            plt.plot(history.history["loss"], label = "train_rmse", marker = "x", color = "blue", alpha = 0.4, linestyle = "-")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()


        return self
    
    
    def predict(self, X):
        return self.model.predict(X)