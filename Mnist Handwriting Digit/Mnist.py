# Import dependencies
import numpy as np 
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helpers import plot
from pre_processing import binary, skeletonize, normalize
from models import Softmax_Regression

# -----------------------LOAD DATASET--------------------------------
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
plot.display_imgs(x_train)

# -----------------------PRE PROCESSING-----------------------------
"""
Pre processing:
    - Binarization
    - Normalization
    - Encoding classes
"""
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
        encoder.fit(y_train.reshape((-1, 1)))
    
    y = encoder.transform(y.reshape((-1, 1))).toarray()
    return normalized, y, encoder, classes


# ---------------------------SOFTMAX REGRESSION-----------------------------
# Train
x_train, y_train, encoder, classes = pre_processing(x_train, y_train)
softmax_regression = Softmax_Regression.SoftmaxRegression(
    input_shape = x_train[0].shape,
    output_shape = len(classes),
    learning_rate = 0.1,
    epochs = 15,
    batchsize = 16
)

softmax_regression.fit(X = x_train, y = y_train)

# Test and evaluate
x_test, y_test, encoder, classes = pre_processing(x_test, y_test, encoder)

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




