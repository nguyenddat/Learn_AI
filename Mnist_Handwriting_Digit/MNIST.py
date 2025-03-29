import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np 
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helpers import plot
from pre_processing import pre_processing

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
plot.display_imgs(x_train)

x_train, y_train, encoder, classes = pre_processing.pre_processing(x_train, y_train)
x_test, y_test, encoder, classes = pre_processing.pre_processing(x_test, y_test, encoder)

# -----------------------SOFTMAX REGRESSION-------------------------
from models import Softmax_Regression

softmax_regression = Softmax_Regression.SoftmaxRegression(
    input_shape = x_train[0].shape,
    output_shape = len(classes),
    learning_rate = 0.1,
    epochs = 15,
    batchsize = 16
)

softmax_regression.fit(X = x_train, y = y_train)

y_predict_prob = softmax_regression.predict(x_test)
y_predict = np.argmax(y_predict_prob, axis = 1)
y_true = np.argmax(y_test, axis = 1)

accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_predict))

conf_matrix = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------------CNN--------------------------------------
from models import Basic_CNN

if len(x_train[0].shape) != 3:
    x_train = np.expand_dims(x_train, axis=-1)

basic_cnn = Basic_CNN.BasicCNN(
    input_shape = x_train[0].shape,
    output_shape = len(classes),
    lr = 0.01,
    epochs = 15,
    batch_size = 16
)

basic_cnn.fit(x_train, y_train)

if len(x_test[0].shape) != 3:
    x_test = np.expand_dims(x_test, axis=-1)

y_predict_prob = basic_cnn.predict(x_test)
y_predict = np.argmax(y_predict_prob, axis = 1)
y_true = np.argmax(y_test, axis = 1)

accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_predict))

conf_matrix = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()