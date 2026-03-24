import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Load data
path = "/home/ibab/ML_data/"
df = pd.read_csv(path + "breast_cancer_dataset.csv")

X = df.drop(["diagnosis", "Unnamed: 32"], axis=1)
y = df["diagnosis"].map({'B': 1, 'M': 0})

# Define model
model = LinearRegression()

# Define KFold
kf = KFold(n_splits=10, shuffle=True, random_state=45)

# Perform cross validation
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