#Implement sigmoid function in python and visualize it
#hypothesis function = g(theta_transpose*x)
import numpy as np
import matplotlib.pyplot as plt

X=[[1,0],[2,1],[3,1],[4,2],[5,3]]
theta=[[0.15],[0.15]]
theta_transpose=np.transpose(theta)
print(theta_transpose)
def Z_func(sample,theta):
    Z_function=0
    for i in range(len(theta)):
        Z_function+=theta[i][0]*sample[i]
    return Z_function

def sigmoid_func(theta,X):
    for i in range(len(X)):
      zee_fun=Z_func(X[i],theta)
      g_zee_fun= 1/(1+np.exp(-zee_fun))
      print("for" ,X[i],"this is the z_function",zee_fun,"and the g function is ,",g_zee_fun)

Sigmoid_function=sigmoid_func(theta,X)
print(Sigmoid_function)
#-----------------------------------------------------------------------------------------------------------------------
#Viewing sigmoid function
Z=[1,2]
def sigmoid_func2(x):
   sigmoid=1/(1+np.exp(-x))
   return sigmoid


x=np.linspace(-10,10,100)
y=sigmoid_func2(x)
plt.plot(x,y)
plt.xlabel('Z value')
plt.ylabel('Sigmoid function')
plt.show()