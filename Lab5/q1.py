#Implement Stochastic Gradient Descent algorithm from scratch
#There are four steps in computing stochastic gradient descent.
#1. Import the sample and choose a random sample.
#2.Compute the hypothesis function and cost function for that sample.(if needed)
#3. Apply the gradient descent on the cost function for that particular sample.
#4.Update the theta values and then choose another random sample and continue the same.
import random

import numpy as np

#Sample
X=[[1,2,5],[3,4,6],[7,8,9]]
Y=[[5],[11],[13]]

theta=[[0],[0],[0]]
learning_rate=0.1
iterations=20

# x_len=len(X)
# K=np.random.randint(0,x_len)
# sample=X[K]
#print("This is my sample",sample)
#Step 1 (hypothesis function)
def hypo_func(theta, sample):
       hypothesis_fun=0
       for i in range(len(theta)):
           hypothesis_fun+=theta[i][0]*sample[i]
       return hypothesis_fun



#Gradient descent for that particular sample

def gradient(theta,sample,Y,K,learning_rate):
    hypo_fun = hypo_func(theta, sample)
    for i in range(len(theta)):
        error=(hypo_fun - Y[K][0])
        theta[i][0]=theta[i][0]-learning_rate*(error*sample[i])
    return theta

def stochastic_gradient(theta):
    for i in range(iterations):


        K = np.random.randint(0, len(X))
        sample = X[K]

        print("Iteration:", i+1)
        print("Theta before updating:", theta)

        theta = gradient(theta, sample, Y, K, learning_rate)

        print("Theta after updating:", theta)
        print("----------------------------------")

    return theta



final_theta = stochastic_gradient(theta)

print("\nFinal Theta:", final_theta)