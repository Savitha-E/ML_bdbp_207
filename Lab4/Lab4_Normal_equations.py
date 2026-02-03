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
x_transpose_into_X=np.multiply(x_transpose,X)
inverse_x_transpose_into_X=np.invert(x_transpose_into_X)
inverse_x_transpose_into_X_into_x_transpose=np.multiply(inverse_x_transpose_into_X,x_transpose)


