#Normal equation:(Linear Regression)
#theta=(X^TX)^-1*X^Ty
import numpy as np

X=[[1,2],[3,4]]
y=[[5],[11]]

# print(X)
# def transpose(X):
#   x_transpose = []
#   for j in range(len(X[0])):
#     new_row=[]
#     for i in range(len(X)):
#         new_row.append(X[i][j])
#     x_transpose.append(new_row)
#   return x_transpose
#
# x_transpose=transpose(X)
# print("This is my X transpose:",x_transpose)
#
# #First compute X_transpose*X
#
# def compute_Xtranspose_multiplied_to_x(X,x_transpose):
#    x_transpose_into_X=[[0,0],[0,0]]
#    for i in range(len(x_transpose)):
#        for j in range(len(X[0])):
#            for k in range(len(X)):
#              x_transpose_into_X[i][j]+=x_transpose[i][k]*X[k][j]
#    return x_transpose_into_X
# x_transpose_X=compute_Xtranspose_multiplied_to_x(X,x_transpose)
# print("This is my X transpose * X:",x_transpose_X)

x_transpose=np.transpose(X)
x_transpose_into_X=np.matmul(x_transpose,X)
inverse_x_transpose_into_X=np.linalg.inv(x_transpose_into_X)
inverse_x_transpose_into_X_into_x_transpose=np.matmul(inverse_x_transpose_into_X,x_transpose)
inverse_x_transpose_into_X_into_x_transpose_into_Y=np.matmul(inverse_x_transpose_into_X_into_x_transpose,y)

print("This is my XT:",x_transpose)
print("This is my X^TX:",x_transpose_into_X)
print("This is my (X^TX)^-1",inverse_x_transpose_into_X)
print("This is my X^TX)^-1*X^T:",inverse_x_transpose_into_X_into_x_transpose)
print("This is my :",inverse_x_transpose_into_X_into_x_transpose_into_Y)
