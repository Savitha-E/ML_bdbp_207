import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    #print(X.shape)
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=99)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(np.mean(X_train_scaled))
    print(np.std(X_train_scaled))
    return X_train_scaled, X_test_scaled

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    mse, r2 = test_model(model, X_test_scaled, y_test)
    print("MSE:", mse)
    print("R2 score is:", r2)

if __name__ == '__main__':
    main()
