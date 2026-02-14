#Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables.
# Compute the gradient of y and print the values.

import numpy as np


def compute_y(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4


def compute_gradient():
    return np.array([2, 3, 3])


points = [
    (1, 2, 3),
    (4, 5, 6),
    (0, 0, 0),
    (2, -1, 3)
]


for point in points:
    grad = compute_gradient()
    print("The point is ,",point,"and its gradient is ,", grad)

