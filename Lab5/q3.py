#Compute the derivative of a sigmoid function and visualize it.
import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-10,10,100)


def sigmoid_func2(x):
   sigmoid=1/(1+np.exp(-x))
   return sigmoid

sigmoid_function=sigmoid_func2(x)

def compute_derivative(sigmoid_function):
    derivative=sigmoid_function*(1-sigmoid_function)
    return derivative

y=compute_derivative(sigmoid_function)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('derivative of sigmoid function')
plt.show()
