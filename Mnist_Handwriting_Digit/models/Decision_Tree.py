import numpy as np

from Mnist_Handwriting_Digit import Multiclass_Classifier


def entropy(y):
    """
    Hàm tính entropy của tập S
    :param:
        - y: np.ndarray
    :return:
        - entropy: float
    """
    H = 0
    for cl in np.unique(y):
        ni = np.sum(y == cl)
        if ni == 0: continue

        pi = ni / y.shape[0]
        H -= pi * np.log2(pi)

    return H


def greddy_search(X, y):
    """
    Tìm kiếm tham lam nhằm tìm feature j và threshold t tối ưu cho cây.
    :param:
        - X: np.ndarray
        - y: np.ndarray
    :return:
        - j: int
        - t: float
    """
    assert X.shape[0] == y.shape[0]
    
    # Initialize
    N, d = X.shape
    j_best, t_best, max_G = None, None, -np.inf
    H_s = entropy(y)

    # Gained Information Function G
    for j in range(d):
        sorted_idx = np.argsort(X[:, j])
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]

        for i in range(1, N):
            t = (X_sorted[i - 1] + X_sorted[i]) / 2

            left_mask = X_sorted[:, j] <= t
            right_mask = X_sorted[:, j] > t

            x_left, x_right = X_sorted[left_mask], X_sorted[right_mask]
            y_left, y_right = y_sorted[left_mask], y_sorted[right_mask]

            H_left = entropy(y_left)
            H_right = entropy(y_right)
            G = H_s - (H_left * x_left.shape[0] / N + H_right * x_right.shape[0] / N)

            if G > max_G:
                j_best, t_best, max_G = j, t, G
                
    return j_best, t_best



class DecisionTree(Multiclass_Classifier.MulticlassClassifer):
    def __init__(self, max_depth = None, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def fit(self, X: np.ndarray, y: np.ndarray, verbose = 1, plot = True):
        self.tree = self.build_tree(X, y, depth = 0, verbose = verbose, plot = plot)
        return self

    
    def build_tree(self, X, y, depth, verbose, plot):
        assert X.shape[0] == y.shape[0]

        N, d = X.shape
        if (self.max_depth is not None) and depth >= self.max_depth:
            return self.create_leaf(y)

        if N <= self.min_samples_split:
            return self.create_leaf(y)
            

        j, t = greddy_search(X, y)
        if j is None or t is None:
            self.create_leaf(y)

        left_mask = X[:, j] <= t
        right_mask = X[:, j] > t

        x_left, x_right = X[left_mask], X[right_mask]
        y_left, y_right = y[left_mask], y[right_mask]

        return {
            "feature": j,
            "threshold": t,
            "left": self.build_tree(x_left, y_left, verbose, plot),
            "right": self.build_tree(x_right, y_right, verbose, plot)
        }


    def create_leaf(self, y):
        classes, counts = np.unique(y, return_counts = True)
        return classes[np.argmax(counts)]
