import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class BasicCNN:
    def __init__(
            self, 
            input_shape, 
            output_shape, 
            lr = 0.1, 
            epochs = 30,
            batch_size = 16
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self._build_model()


    def _build_model(self):
        input = tf.keras.layers.Input(self.input_shape)

        x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = "relu")(input)
        x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)

        x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation = "relu")(x)

        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(self.output_shape, activation = "softmax")(x)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

        self.model.summary()
        return self
    

    def fit(self, X, y, verbose = 1):
        history = self.model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, verbose = verbose)
        plt.plot(history.history["loss"], label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Over Epochs")
        plt.show()

        return self


    def predict(self, x):
        return self.model.predict(x)