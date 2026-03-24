#Lab 6 - k-fold Cross Validation & Model Selection
#K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.

'''
Steps
1.Load data.
2.Split the data into k parts .
3.Apply the training model for each k part .
4.Get the average accuracy by adding k accuracy results .
5.Find the standard deviation for the final accuracy .
'''
import numpy as np
import pandas as pd
from numpy.ma.core import concatenate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
path="/home/ibab/ML_data/"
df=pd.read_csv(path+"breast_cancer_dataset.csv")
print(df.shape)
print(df.columns)
k=10
def load_data ( df):
    X=df.drop(["diagnosis","Unnamed: 32"], axis=1)
    Y=df["diagnosis"]
    y = Y.map({'B': 1, 'M': 0})
    return X,y

X,y=load_data (df)
print ("This is X,",X)
print("This is y," ,y)

def split_into_kfolds(X,y,k):
   x_split= np.array_split(X,k,axis=0)
   y_split= np.array_split(y,k,axis=0)
   return x_split,y_split

def split_data(X, y):
    return train_test_split(X, y, test_size=0.1, random_state=99)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def main():
    X, y = load_data(df)
    X_folds, y_folds = split_into_kfolds(X, y, k)

    all_mse = []
    all_r2 = []

    for i in range(k):


        X_test = X_folds[i]
        y_test = y_folds[i]


        X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(k) if j != i])


        X_train, X_test = scale_data(X_train, X_test)


        model = LinearRegression()
        model.fit(X_train, y_train)


        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)

        print(f"Fold {i+1}")
        print("MSE:", mse)
        print("R2:", r2)
        print("---------------")

    print("Final Results")
    print("Average MSE:", np.mean(all_mse))
    print("Std Dev MSE:", np.std(all_mse))
    print("Average R2:", np.mean(all_r2))
    print("Std Dev R2:", np.std(all_r2))


if __name__ == '__main__':
    main()



