# Use the above simulated CSV file and implement the following from scratch in Python
# Read simulated data csv file
# Form x and y (disease_score_fluct)
# Write a function to compute hypothesis
# Write a function to compute the cost
# Write a function to compute the derivative
# Write update parameters logic in the main function

import numpy as np
import pandas as pd


#1.Read simulated data csv file,Form x and y (disease_score_fluct)
def load_data():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop("disease_score_fluct", axis=1)
    y = df["disease_score_fluct"]
    return X,y


# print(load_data())

#2.Form x and y ----------------------------------------------------------------------------------------------------
def get_Y():
    X,y = load_data()
    return y

def get_X():
    X,y = load_data()
    return X

# print(get_Y())
# print(get_X())
#
x_values=get_X()
y_values=get_Y()


X=np.array(x_values)
y=np.array(y_values)
#print(X)

def get_no_theta():
    X=np.array(x_values)
    theta=[ ]
    for j in range(len(X[0])):
      theta.append([0])
    return theta

theta=(get_no_theta())
# print(theta)
# print(X)
# print(y)
#3.Form hypothesis function ----------------------------------------------------------------------------------------------------

#for j in range(len(theta[0])):
# print(np.shape(X))
# print(np.shape(theta))

# X_theta=np.dot(X,theta)
# # print(X_theta)
# theta_0=[[0]]
# X_theta=np.vstack((X_theta,theta_0))
# print(X_theta)
# print(np.shape(X_theta))

def hypothesis(X,theta):
    X_theta=np.dot(X,theta)
    theta_zero=[[0]]
    X_theta=np.vstack((X_theta,theta_zero))
    return X_theta

X_theta=hypothesis(X,theta)
print(X_theta)
print(np.shape(X_theta))

#4.Form Cost function---------------------------------------------------------------------------------------------------------








