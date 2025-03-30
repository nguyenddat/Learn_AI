import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


from Regression_Model import RegressionModel

class PolynomialRegression(RegressionModel):
    def __init__(self, order, epochs = 10, lr = 0.1, batch_size = 16):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.poly_scaler = PolynomialFeatures(degree = order)


    def _build_model(self, input_shape, output_shape):
        input = tf.keras.layers.Input(shape = input_shape)

        output = tf.keras.layers.Dense(units = output_shape, activation = "linear")(input)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr),
            loss = "mse",
            metrics = [tf.keras.metrics.RootMeanSquaredError()]
        )
        self.model.summary()
        return self


    def fit(self, X, y, plot = True):
        poly_x = self.poly_scaler.fit_transform(X)
        self._build_model(input_shape = poly_x[0].shape, output_shape = 1)
        history = self.model.fit(
            x = poly_x,
            y = y,
            epochs = self.epochs,
            batch_size = self.batch_size,
            verbose = 1,
            validation_split = 0.2
        )

        if plot:
            metrics = ["loss", "root_mean_squared_error", "val_loss", "val_root_mean_squared_error"]
            fig, axes = plt.subplots(2, 2, figsize = (20, 10))

            for y in range(2):
                for x in range(2):
                    axe = axes[y, x]
                    axe.plot(history.history[metrics[y * 2 + x]], label = metrics[y * 2 + x], color = "blue", alpha = 0.4, linestyle = "-")
                    axe.set_xlabel("Epochs")
                    axe.set_ylabel(f"{metrics[y * 2 + x]}")
            
            plt.tight_layout()
            plt.show()

        return self

    def predict(self, X):
        poly_x = self.poly_scaler.transform(X)
        return self.model.predict(poly_x)
