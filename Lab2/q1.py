#x matrix
X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

n = len(X)
m = len(X[0])


means = []
for j in range(m):
    col_sum = 0
    for i in range(n):
        col_sum += X[i][j]
    means.append(col_sum / n)

print("Mean of the columns are,", means)



X_centered = []
for i in range(n):
    row = []
    for j in range(m):
        row.append(X[i][j] - means[j])
    X_centered.append(row)


cov_matrix = []

for i in range(m):
    cov_row = []
    for j in range(m):
        value = 0
        for k in range(n):
            value += X_centered[k][i] * X_centered[k][j]
        cov_row.append(value / (n - 1))
    cov_matrix.append(cov_row)

print("The Covariance Matrix is :")
for row in cov_matrix:
    print(row)
