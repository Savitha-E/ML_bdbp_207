#Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.

from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data ( ):
    df=pd.read_csv("/home/ibab/ML_data/breast_cancer_dataset.csv")
    X=df.drop(["diagnosis","Unnamed: 32"], axis=1)
    Y=df["diagnosis"]
    y = Y.map({'B': 1, 'M': 0})
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def test_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return score,r2

def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return mse,r2


if __name__ == "__main__":
    main()




