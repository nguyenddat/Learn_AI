from abc import ABC, abstractmethod

import numpy as np


class RegressionModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError()
    
    
    @abstractmethod
    def predict(self, X: np.ndarray):
        raise NotImplementedError