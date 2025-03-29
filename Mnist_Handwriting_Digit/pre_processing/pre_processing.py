import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

from . import binary, normalize

def pre_processing(X, y, encoder = None):
    # Binarization
    binarized = np.array([
        binary.binary(img) for img in tqdm(X, desc = "Binarying...")
    ])

    # Normalization
    normalized = np.array([
        normalize.normalize(img) for img in tqdm(X, desc = "Normalizing...")
    ])
    
    # Classes
    classes = np.unique(y)

    # Encoding classes
    if encoder is None:
        encoder = OneHotEncoder()
        encoder.fit(y.reshape((-1, 1)))
    
    y = encoder.transform(y.reshape((-1, 1))).toarray()
    return normalized, y, encoder, classes