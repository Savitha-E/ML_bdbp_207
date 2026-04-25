import numpy as np


# --------------------------------
# Transform Function
# --------------------------------
def transform(x):
    x1 = x[0]
    x2 = x[1]

    phi = np.array([
        x1**2,
        np.sqrt(2) * x1 * x2,
        x2**2
    ])

    return phi


# --------------------------------
# Polynomial Kernel
# --------------------------------
def polynomial_kernel(a, b):
    return (np.dot(a, b))**2


# --------------------------------
# Main Example
# --------------------------------
# Given vectors
x1 = np.array([3, 6])
x2 = np.array([10, 10])


# Step 1: Transform both points
phi_x1 = transform(x1)
phi_x2 = transform(x2)

# Step 2: Dot product in higher dimension
dot_product_high = np.dot(phi_x1, phi_x2)

# Step 3: Kernel (without transform)
kernel_value = polynomial_kernel(x1, x2)


# --------------------------------
# Output
# --------------------------------
print("Transformed x1:", phi_x1)
print("Transformed x2:", phi_x2)

print("\nDot product in higher dimension:", dot_product_high)

print("Kernel value (without transform):", kernel_value)