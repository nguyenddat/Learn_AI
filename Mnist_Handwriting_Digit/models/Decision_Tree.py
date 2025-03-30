from typing import *

import numpy as np
from tqdm import tqdm

from helpers.model_helpers import *


class DecisionTree:
    def __init__(
            self,
            min_samples: int = 5,
            max_depth: int = 10,
    ):
        """
        :param
            - min_samples: số observations tối thiểu để chia cành
            - max_depth: độ sâu tối đa của cây
        """
        self.min_samples = min_samples
        self.max_depth = max_depth


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            verbose: int = 1,
            current_depth: int = 0
    ):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        assert X.shape[0] == y.shape[0]
        N, d = X.shape

        # Nếu có thể tách node ==> tìm node trái phải
        if (N >= self.min_samples) and (current_depth <= self.max_depth):
            best_split = self.best_split(X, y)

            if best_split["gain"] != 0:
                if verbose:
                    print(f"""Depth: {current_depth} --> Gain: {best_split["gain"]}""")

                left_node = self.fit(best_split["left_observations"], best_split["left_labels"], verbose, current_depth + 1)
                right_node = self.fit(best_split["right_observations"], best_split["right_labels"], verbose, current_depth + 1)


                return Node(
                    feature = best_split["feature"],
                    threshold = best_split["threshold"],
                    left = left_node,
                    right = right_node,
                    gain = best_split["gain"]
                )
        
        # Nếu không, trả về value
        leaf_value = self.leaf_value(y)
        return Node(value = leaf_value)
    

    def leaf_value(self, y):
        unique_labels, counts = np.unique(y, return_counts = True)
        return unique_labels[np.argmax(counts)]
    

    def best_split(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        N, d = X.shape
        best_split = {
            "gain": 0,
            "feature": None,
            "threshold": None,            
        }

        # Duyệt toàn bộ các feature
        for j in tqdm(range(d), desc = "Finding t, j..."):
            unique_xj = np.unique(X[:, j])
            
            for t in unique_xj:
                gain = information_gain(j, t, X, y)
                
                if gain["gain"] > best_split["gain"]:
                    best_split["gain"] = gain["gain"]
                    best_split["feature"] = j
                    best_split["threshold"] = t
                    best_split["left_observations"] = gain["left_observations"]
                    best_split["right_observations"] = gain["right_observations"]
                    best_split["left_labels"] = gain["left_labels"]
                    best_split["right_labels"] = gain["right_labels"]
        
        return best_split


class Node:
    """
    Biểu diễn Node trong cây
    """
    def __init__(
            self,
            feature: int = None,
            threshold: float = None,
            left: Any = None,
            right: Any = None,
            gain: float = None,
            value: Any = None
    ):
        """
        :param
            - feature: feature thứ j
            - threshold: ngưỡng t
            - left: left child node
            - right: right child node
            - gain: information gain
            - value: Nếu đây là lớp lá ==> đây là giá trị của nhãn dự đoán
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

