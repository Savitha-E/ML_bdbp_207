#Implementtaion of k means from scratch for an example from Previous year QP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# Load Data
# --------------------------------
def load_data():
    data = {
        "X1": [1, 1, 0, 5, 6, 4],
        "X2": [4, 3, 4, 1, 2, 0]
    }
    df = pd.DataFrame(data)
    return df


df = load_data()
X = df.values

print("Dataset:")
print(df)


# --------------------------------
# Plot observations
# --------------------------------
def plot_points(df, title="Original Data", labels=None, centroids=None):
    plt.figure(figsize=(6, 5))

    if labels is None:
        plt.scatter(df["X1"], df["X2"], s=100)
    else:
        for i in range(len(df)):
            plt.scatter(df["X1"][i], df["X2"][i], c=f"C{labels[i]}", s=100)

    for i in range(len(df)):
        plt.text(df["X1"][i] + 0.05, df["X2"][i] + 0.05, str(i + 1))

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=250, marker="X", color="black", label="Centroids")
        plt.legend()

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.grid(True)
    plt.show()


plot_points(df, title="Original Observations")


# --------------------------------
# Random initial cluster assignment
# --------------------------------
def initial_clusters(X, K=2):
    np.random.seed(42)
    clusters = np.random.choice(K, size=len(X))
    print("\nInitial Cluster Labels:")
    print(clusters)
    return clusters


# --------------------------------
# Compute centroids
# --------------------------------
def compute_centroids(X, clusters, K=2):
    centroids = []

    for k in range(K):
        points = X[clusters == k]

        if len(points) == 0:
            centroid = np.zeros(X.shape[1])
        else:
            centroid = np.mean(points, axis=0)

        centroids.append(centroid)

    centroids = np.array(centroids)

    print("\nCentroids:")
    print(centroids)

    return centroids


# --------------------------------
# Assign clusters using Euclidean distance
# --------------------------------
def assign_clusters(X, centroids):
    new_clusters = []

    for point in X:
        distances = []

        for centroid in centroids:
            distance = np.linalg.norm(point - centroid)
            distances.append(distance)

        nearest_cluster = np.argmin(distances)
        new_clusters.append(nearest_cluster)

        print("Point:", point, "Distances:", distances, "Assigned Cluster:", nearest_cluster)

    new_clusters = np.array(new_clusters)

    print("\nUpdated Cluster Labels:")
    print(new_clusters)

    return new_clusters


# --------------------------------
# K-Means algorithm
# --------------------------------
def kmeans(X, K=2, iterations=10):
    clusters = initial_clusters(X, K)

    for i in range(iterations):
        print("\n---------------------------")
        print("Iteration", i + 1)
        print("---------------------------")

        centroids = compute_centroids(X, clusters, K)
        new_clusters = assign_clusters(X, centroids)

        if np.array_equal(clusters, new_clusters):
            print("\nConverged!")
            break

        clusters = new_clusters

    centroids = compute_centroids(X, clusters, K)

    return clusters, centroids


final_clusters, final_centroids = kmeans(X, K=2)

print("\nFinal Cluster Labels:")
print(final_clusters)

print("\nFinal Centroids:")
print(final_centroids)


# --------------------------------
# Plot final clusters
# --------------------------------
plot_points(
    df,
    title="Final K-Means Clusters",
    labels=final_clusters,
    centroids=final_centroids
)