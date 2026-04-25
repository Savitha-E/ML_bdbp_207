import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------------
# Load Data
# --------------------------------
def load_data():
    data = {
        "x1": [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
        "x2": [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
        "Label": [
            "Blue", "Blue", "Red", "Red", "Red",
            "Blue", "Red", "Red", "Blue", "Red",
            "Red", "Red", "Blue", "Blue", "Blue"
        ]
    }

    df = pd.DataFrame(data)
    return df


# --------------------------------
# RBF Kernel Formula
# --------------------------------
def rbf_kernel(point1, point2, gamma=0.1):
    distance_squared = np.sum((point1 - point2) ** 2)
    similarity = np.exp(-gamma * distance_squared)
    return similarity


# --------------------------------
# Create RBF Kernel Matrix
# --------------------------------
def create_rbf_matrix(X, gamma=0.1):
    n = len(X)

    kernel_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            kernel_matrix[i, j] = rbf_kernel(X[i], X[j], gamma)

    return kernel_matrix


# --------------------------------
# Plot Data
# --------------------------------
def plot_data(df):
    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        plt.scatter(subset["x1"], subset["x2"], label=label, s=100)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original Data")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------
# Train SVM using RBF Kernel
# --------------------------------
def train_svm(X, y):
    model = SVC(kernel="rbf", gamma=0.1, C=1)
    model.fit(X, y)
    return model


# --------------------------------
# Main Program
# --------------------------------
df = load_data()

print("Dataset:")
print(df)

X = df[["x1", "x2"]].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Label"])

print("\nEncoded Labels:")
print(y)

plot_data(df)

kernel_matrix = create_rbf_matrix(X, gamma=0.1)

print("\nRBF Kernel Matrix:")
print(np.round(kernel_matrix, 3))

model = train_svm(X, y)

y_pred = model.predict(X)

print("\nPredicted Labels:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))