
#Excercise from ISLP textbook 12.6 section

#Q6

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
# Method 1: PVE using sklearn
# --------------------------------
def pve_using_sklearn(pca):
    return pca.explained_variance_ratio_


# --------------------------------
# Method 2: Manual PVE calculation
# --------------------------------
def pve_manual(scaled_data, scores):
    total_variance = np.sum(scaled_data ** 2)

    pve_list = []

    for i in range(scores.shape[1]):
        pc_variance = np.sum(scores[:, i] ** 2)
        pve = pc_variance / total_variance
        pve_list.append(pve)

    return np.array(pve_list)


# --------------------------------
# Main Program
# --------------------------------
data = load_data_set()

scaled_data = scale_data(data)

pca, scores = apply_pca(scaled_data)


pve1 = pve_using_sklearn(pca)


pve2 = pve_manual(scaled_data, scores)


print("PVE using sklearn:")
print(pve1)

print("\nPVE using manual formula:")
print(pve2)