import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

from pre_processing import normalize

mnist = tf.keras.datasets.california_housing
(x_train, y_train),(x_test, y_test) = mnist.load_data()

features = {
    "MedInc": "median income in block group",
    "HouseAge": "median house age in block group",
    "AveRooms": "average number of rooms per household",
    "AveBedrms": "average number of bedrooms per household",
    "Population": "block group population",
    "AveOccup": "average number of household members",
    "Latitude": "block group latitude",
    "Longitude": "block group longitude"
}

normalized_X_train, x_scaler = normalize.normalize(x_train)
normalized_X_test = normalize.normalize(x_test, x_scaler)

normalized_y_train, y_scaler = normalize.normalize(y_train.reshape(-1, 1))
normalized_y_test = normalize.normalize(y_test.reshape(-1, 1), y_scaler)


# -------------------------ANN-----------------------------------
from models.ANN import ANN

ann_model = ANN(input_shape = normalized_X_train[0].shape, output_shape = 1)
ann_model.fit(X = normalized_X_train, y = normalized_y_train)

# -----------------------Linear Regression-----------------------
from models.Linear_Regression import LinearRegression

linear_regression_model = LinearRegression(input_shape = normalized_X_train[0].shape, output_shape = 1)
linear_regression_model.fit(X = normalized_X_train, y = normalized_y_train)

