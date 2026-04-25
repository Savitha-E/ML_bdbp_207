#Excercise from ISLP textbook 12.6 section

#Q6

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# --------------------------------
# Load Data
# --------------------------------
def load_data_set():
    data = pd.read_csv("USArrests.csv", index_col=0)
    return data


# --------------------------------
# Scale Data
# --------------------------------
def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# --------------------------------
# Apply PCA
# --------------------------------
def apply_pca(scaled_data):
    pca = PCA()
    scores = pca.fit_transform(scaled_data)
    return pca, scores


# --------------------------------
# Get PCA Loadings using Regression
# --------------------------------
def get_loadings_by_regression(scaled_data, scores, M=2):
    Z = scores[:, :M]

    loadings = []

    for j in range(scaled_data.shape[1]):
        y = scaled_data[:, j]

        model = LinearRegression(fit_intercept=False)
        model.fit(Z, y)

        loadings.append(model.coef_)

    loadings = np.array(loadings)
    return loadings


# --------------------------------
# Main Program
# --------------------------------
data = load_data_set()

print("First 5 rows of data:")
print(data.head())

scaled_data = scale_data(data)

pca, scores = apply_pca(scaled_data)

regression_loadings = get_loadings_by_regression(scaled_data, scores, M=2)

print("\nLoadings using regression:")
print(regression_loadings)

print("\nPCA loadings from sklearn:")
print(pca.components_[:2].T)