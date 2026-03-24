# Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression.
# SONAR dataset is a binary classification problem with target variables as Metal or Rock. i.e.
# signals are from metal or rock.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split , KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def LoadData():
  path="/home/ibab/ML_data/"
  df=pd.read_csv(path+"sonar.all-data.csv")
  X=df.drop(columns=['R'])
  Y=df['R']
  y=Y.map({"R":1,"M":0})
  return X,y
X,y=LoadData()
model = LogisticRegression()


kf = KFold(n_splits=10, shuffle=True, random_state=42)


mse_scores = -cross_val_score(model, X, y,
                              cv=kf,
                              scoring='neg_mean_squared_error')

r2_scores = cross_val_score(model, X, y,
                            cv=kf,
                            scoring='r2')

print("Average MSE:", np.mean(mse_scores))
print("Std MSE:", np.std(mse_scores))
print("Average R2:", np.mean(r2_scores))
print("Std R2:", np.std(r2_scores))



