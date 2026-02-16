import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop("disease_score", axis=1).values
    y = df["disease_score"].values
    return X, y

def X0_column(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)

def normal_equation(X, y):
    X_transpose = np.transpose(X)
    XTX = np.matmul(X_transpose, X)
    XTX_inverse = np.linalg.inv(XTX)
    XTX_inverse_XT = np.matmul(XTX_inverse, X_transpose)
    theta = np.matmul(XTX_inverse_XT, y)
    return theta

def predict(X, theta):
    return np.matmul(X, theta)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )
    X_train = X0_column(X_train)
    X_test = X0_column(X_test)
    theta = normal_equation(X_train, y_train)
    y_pred = predict(X_test, theta)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Theta values using my Normal Equation model:", theta)
    print("MSE:", mse)
    print("R2 Score:", r2)

if __name__ == "__main__":
    main()
