import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop("disease_score", axis=1).values
    y = df["disease_score"].values
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Using scikit normal equations model-----------------")
    print("MSE:", mse)
    print("R2 Score:", r2)

if __name__ == "__main__":
    main()
