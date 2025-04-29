from typing import *

class ScratchedLinearRegression:
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

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        pass

    def mae(self, y_true, y_pred):
        pass

    def _build_model(self):
        pass