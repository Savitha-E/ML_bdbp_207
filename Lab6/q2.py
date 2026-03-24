#Data normalization - scale the values between 0 and 1. Implement code from scratch.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('/home/ibab/ML_data/breast_cancer_dataset.csv')
    X=df.drop(['diagnosis','Unnamed: 32'],axis=1)
    Y=df['diagnosis']
    y=Y.map({'B':1,'M':0})
    return X

X=load_data()

# print(len(X.columns))
# print(X.shape)
# print(X['id'])


# def ran(X):
#     count = 0
#     for i in X.columns:
#         print(X[i])
#         count += 1
#     return count
#
# print(ran(X))

def normalizing(X):
    for i in X.columns:
        col_max = np.max(X[i])
        col_min = np.min(X[i])
        for idx in X.index:
            X.loc[idx, i] = (X.loc[idx, i] - col_min) / (col_max - col_min)
    return X

print(normalizing(X))
print(X)
