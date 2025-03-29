from typing import *

import tensorflow as tf
import matplotlib.pyplot as plt

from Mnist_Handwriting_Digit.helpers import plot
from Mnist_Handwriting_Digit import Multiclass_Classifier

class SoftmaxRegression(Multiclass_Classifier.MulticlassClassifer):
    def __init__(
            self,
            input_shape: tuple,
            output_shape: int,
            learning_rate: float = 0.1,
            epochs: int = 1000,
            batchsize: int = 16
    ):
        """
        Linear Regression Model from tensorflow
        :param:
            - input_shape: int
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batchsize = batchsize

        self._build_model()
    

    def _build_model(self):
        input = tf.keras.layers.Input(shape = self.input_shape)

        x = tf.keras.layers.Flatten()(input)
        
        output = tf.keras.layers.Dense(units = self.output_shape, activation = "softmax")(x)

        self.model = tf.keras.Model(inputs = input, outputs = output)
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )
        
        self.model.summary()
        return self


    def fit(self, X, y, verbose = 1, plot = True):
        """
        Fit the model on the data
        :param:
            - X: np.ndarray
            - y: np.ndarray"
            - epochs: int
        """
        history = self.model.fit(
            x = X, 
            y = y,
            validation_split = 0.2, 
            epochs = self.epochs,
            batch_size = self.batchsize, 
            verbose = verbose
        )

        if plot:
            plot.plot_history(history)

        return self


    def predict(self, X):
        """
        Predict the output
        :param:
            - X: np.ndarray
        """
        return self.model.predict(X)
