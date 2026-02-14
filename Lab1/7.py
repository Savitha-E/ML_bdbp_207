
theta = [[2],
         [3],
         [3]]


X = [[1, 0, 2],
     [0, 1, 1],
     [2, 1, 0],
     [1, 1, 1],
     [0, 2, 1]]


def matrix_multiply(X, theta):
    result = []

    for i in range(len(X)):
        row_result = 0
        for j in range(len(theta)):
            row_result += X[i][j] * theta[j][0]
        result.append([row_result])

    return result



output = matrix_multiply(X, theta)

print("Xθ =")
for value in output:
    print(value)
