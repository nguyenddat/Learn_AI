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
        return self.model.predict(X)