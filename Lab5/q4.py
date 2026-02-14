#Implement logistic regression using scikit-learn for the breast cancer dataset -
"""
1.first load data
2.split the data into x,y
3.split it further into x-train,x-test,y-train,y-test
4.scale the data
5.train the data
6.predict y
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    path = "/home/ibab/ML_data/"
    df = pd.read_csv(path + "breast_cancer_dataset.csv")

    # Drop unnecessary columns
    X = df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)


    y = df['diagnosis'].map({"M": 1, "B": 0})

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    return accuracy,y_pred

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    accuracy= test_model(model, X_test_scaled, y_test)

    print("The Accuracy and predicted y value is :", accuracy)

if __name__ == '__main__':
    main()
