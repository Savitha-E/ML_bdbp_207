import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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

    labels = np.array([0]*20 + [1]*20 + [2]*20)

    return X, labels


# --------------------------------
# Apply PCA
# --------------------------------
def apply_pca(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# --------------------------------
# Plot PCA
# --------------------------------
def plot_pca(X_pca, labels):
    plt.figure(figsize=(7, 5))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=80)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Plot of Simulated Data")
    plt.grid(True)
    plt.show()


# --------------------------------
# Main Program
# --------------------------------
X, labels = create_data()

pca, X_pca = apply_pca(X)

print("Original data shape:")
print(X.shape)

print("\nPCA data shape:")
print(X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

plot_pca(X_pca, labels)