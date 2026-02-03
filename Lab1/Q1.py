#Implement A transpose *A  ,  A = [[1 2 3],[4 5 6]]

A=[[1,2,3],[4,5,6]]


def transpose(A):
  for j in range(len(A[0])):
      for i in range(len(A)):
          print(A[i][j],end=" ")
      print()

print(transpose(A))


# for j in range(len(A[0])):
#     for i in range(len(A)):
#         print(A[i][j], end=" ")
#     print()
