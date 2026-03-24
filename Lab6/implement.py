#Linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

def load_data():
    path = "/home/ibab/ML_data/"
    df = pd.read_csv(path + "breast_cancer_dataset.csv")
    X = df.drop(["diagnosis", "Unnamed: 32"], axis=1)
    Y = df["diagnosis"]
    y=Y.map({"B":1,"M":0})
    return X, y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train,X_test):
    scalar= StandardScaler()
    xtrain_scaled=scalar.fit_transform(X_train)
    xtest_scaled=scalar.transform(X_test)
    return xtrain_scaled, xtest_scaled

def test_model(X_test,y_test,model,mean_squared_error,r2_score):
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    return mse,r2

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse, r2 = test_model(X_test,y_test,model,mean_squared_error,r2_score)
    print("The mean squared error is",mse)
    print("The r2 score is",r2)

if __name__ == "__main__":
    main()





