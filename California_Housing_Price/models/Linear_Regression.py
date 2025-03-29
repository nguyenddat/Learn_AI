import tensorflow as tf

import matplotlib.pyplot as plt
from Regression_Model import RegressionModel


class LinearRegression(RegressionModel):
    def __init__(self, input_shape, output_shape, epochs = 30, lr = 0.01, batch_size = 16):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self._build_model()

    
    def _build_model(self):
        input = tf.keras.layers.Input(shape = self.input_shape)

        output = tf.keras.layers.Dense(units = self.output_shape)(input)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr),
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