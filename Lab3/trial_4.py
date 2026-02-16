import numpy as np



x1=[[1],[1 ],[2]]
x2=[[1],[3 ],[4 ]]
y=[[5],[11]]
x1_transpose=np.transpose(x1)
x2_transpose=np.transpose(x2)
# print(x1_transpose)
# print(x2_transpose)
X=np.vstack((x1_transpose,x2_transpose))
print(np.shape(X))

#---------------------
def theta(X):
    theta1 = []
    for j in range(len(X[0])):
        X=np.array(X)
        theta1.append([0])
    return(theta1)

theta1=theta(X)

#-------------------------
#hypothesis function
def hypothesis_function(X,theta1):
    hypothesis_fun=np.dot(X,theta1)
    return hypothesis_fun
hypothesis_fun=hypothesis_function(X,theta1)
#------------------
#cost function
def cost_function(y,hypothesis_fun):
   summation=np.sum((hypothesis_fun-y)**2)
   cost=0.5*summation
   return cost
cost_fun=cost_function(y,hypothesis_fun)
#------------------
#derivative function



#-------------------
print("This is X",X)
print("This is Y",y)
print("This is my theta1",theta1)
print("This is my hypothesis function",hypothesis_function(X,theta1))
print("This is my cost_fun",cost_fun)
