import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# --------------------------------
# Create simulated data
# --------------------------------
def create_data():
    np.random.seed(0)

    n = 20
    p = 50

    class1 = np.random.normal(0, 1, (n, p))
    class2 = np.random.normal(5, 1, (n, p))
    class3 = np.random.normal(10, 1, (n, p))

    X = np.vstack([class1, class2, class3])

    true_labels = np.array([0]*20 + [1]*20 + [2]*20)

    return X, true_labels


# --------------------------------
# Apply PCA
# --------------------------------
def apply_pca(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return X_pca


# --------------------------------
# Apply K-means on PCA scores
# --------------------------------
def apply_kmeans(X_pca):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    return clusters


# --------------------------------
# Compare true labels and clusters
# --------------------------------
def compare_labels(true_labels, clusters):
    table = pd.crosstab(true_labels, clusters)
    return table


# --------------------------------
# Plot K-means clusters
# --------------------------------
def plot_clusters(X_pca, clusters):
    plt.figure(figsize=(7, 5))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=80)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-means Clustering on First Two PCA Scores")
    plt.grid(True)
    plt.show()


# --------------------------------
# Main Program
# --------------------------------
X, true_labels = create_data()

X_pca = apply_pca(X)

clusters = apply_kmeans(X_pca)

table = compare_labels(true_labels, clusters)

print("K-means cluster labels:")
print(clusters)

print("\nComparison between true labels and clusters:")
print(table)

plot_clusters(X_pca, clusters)