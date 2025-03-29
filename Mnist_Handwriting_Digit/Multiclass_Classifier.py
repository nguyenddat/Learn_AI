from abc import ABC, abstractmethod

import numpy as np


class MulticlassClassifer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: int, plot_history: bool):
        raise NotImplementedError()
    
    
    @abstractmethod
    def predict(self, X: np.ndarray):
        raise NotImplementedError()
        