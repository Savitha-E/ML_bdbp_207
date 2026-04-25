import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------
# Load Data (from your table)
# --------------------------------
def load_data():
    data = {
        "x1": [1,1,2,3,6,9,13,18,3,6,6,9,10,11,12,16],
        "x2": [13,18,9,6,3,2,1,1,15,6,11,5,10,5,6,3],
        "Label": ["Blue","Blue","Blue","Blue","Blue","Blue","Blue","Blue",
                  "Red","Red","Red","Red","Red","Red","Red","Red"]
    }
    return pd.DataFrame(data)


# --------------------------------
# Transform Function
# --------------------------------
def transform(X):
    transformed = []

    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]

        phi = [
            x1**2,
            np.sqrt(2) * x1 * x2,
            x2**2
        ]

        transformed.append(phi)

    return np.array(transformed)


# --------------------------------
# Plot Original Data (2D)
# --------------------------------
def plot_2d(df):
    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        plt.scatter(subset["x1"], subset["x2"], label=label)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original Data (2D)")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------
# Plot Transformed Data (3D)
# --------------------------------
def plot_3d(X_transformed, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X_transformed)):
        if labels[i] == "Blue":
            color = "blue"
        else:
            color = "red"

        ax.scatter(
            X_transformed[i][0],
            X_transformed[i][1],
            X_transformed[i][2],
            color=color
        )

    ax.set_xlabel("x1^2")
    ax.set_ylabel("√2 x1x2")
    ax.set_zlabel("x2^2")
    ax.set_title("Transformed Data (3D)")
    plt.show()


# --------------------------------
# Dot Product in Higher Dimension
# --------------------------------
def dot_product_example():
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    phi_x1 = transform([x1])[0]
    phi_x2 = transform([x2])[0]

    dot_product = np.dot(phi_x1, phi_x2)

    print("\nTransformed x1:", phi_x1)
    print("Transformed x2:", phi_x2)
    print("Dot product in higher dimension:", dot_product)


# --------------------------------
# Polynomial Kernel
# --------------------------------
def polynomial_kernel(a, b):
    return (np.dot(a, b))**2


def kernel_example():
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    kernel_value = polynomial_kernel(x1, x2)

    print("\nKernel value (without transform):", kernel_value)


# --------------------------------
# MAIN
# --------------------------------
df = load_data()

X = df[["x1", "x2"]].values
labels = df["Label"].values

# Plot original
plot_2d(df)

# Transform
X_transformed = transform(X)

print("\nTransformed Data (first 5 rows):")
print(X_transformed[:5])

# Plot transformed
plot_3d(X_transformed, labels)

# Dot product check
dot_product_example()

# Kernel trick
kernel_example()