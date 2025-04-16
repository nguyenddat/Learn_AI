from typing import *

import tensorflow as tf
import matplotlib.pyplot as plt

class MultiLogisticRegression:
    def __init__(
            self,
            input_shape: Tuple[int, int],
            lr: float = 0.1,
            epochs: int = 100,
            batch_size: int = 16,

    ):
        self.input_shape = input_shape
        self.output_shape = 1
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._build_model()
    
    def _build_model(self):
        input = tf.keras.layers.Input(shape = self.input_shape)

        dense1 = tf.keras.layers.Dense(units = 16, activation = "relu")(input)
        dense2 = tf.keras.layers.Dense(units = 8, activation = "relu")(dense1)

        output = tf.keras.layers.Dense(
            units = self.output_shape,
            activation = "sigmoid"
        )(dense2)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr),
            loss = tf.losses.BinaryCrossentropy(),
            metrics = ["accuracy"]
        )
        self.model.summary()

        return self


    def fit(self, X, y):
        history = self.model.fit(
            x = X,
            y = y,
            batch_size = self.batch_size,
            epochs = self.epochs,
            verbose = 1,
            validation_split = 0.2,
        )
        
        hist = history.history

        # Plot Loss
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(hist['loss'], label='Training Loss')
        plt.plot(hist['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy (nếu dùng metric này)
        if 'accuracy' in hist:
            plt.subplot(1, 2, 2)
            plt.plot(hist['accuracy'], label='Training Accuracy')
            plt.plot(hist['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.show()        
        return self
    
    
    def predict(self, x):
        return self.model.predict(x)