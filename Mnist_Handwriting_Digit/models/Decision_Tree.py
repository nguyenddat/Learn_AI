import numpy as np

def entropy(X, y):
    """
    Hàm tính entropy của tập S
    :param:
        - X: np.ndarray
        - y: np.ndarray
    :return:
        - entropy: float
    """
    assert X.shape[0] == y.shape[0]

    # N = |S|
    N = X.shape[0]

    # C: unique classes
    C = np.unique(y)

    # Entropy
    H = 0

    for i in C:
        """
        pi: P(y = cls)
        """
        Ni = np.sum(y == i)
        pi = Ni / N

        H += (pi * np.log2(pi))

    return -H



class DecisionTree:
    def __init__(self):
        pass

    
    
    def fit(self):
        pass