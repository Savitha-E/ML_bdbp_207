

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    df=pd.read_csv("/home/ibab/ML_data/breast_cancer_dataset.csv")
    X=df.drop(["diagnosis","Unnamed: 32"],axis=1)
    Y=df["diagnosis"]
    y=Y.map({"B":0,"M":1})
    return X,y

def split_data(X,y):
  return train_test_split(X,y,test_size=0.3,random_state=99)

def train_model(X_train,Y_train):
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train,Y_train)
   return model

def test_model(model,X_test,Y_test):
 y_pred= model.predict(X_test)
 accuracy = accuracy_score(Y_test,y_pred)
 return accuracy

def main():
    X,y=load_data()
    X_train, X_test,y_train,y_test = split_data(X,y)
    model = train_model(X_train,y_train)
    accuracy = test_model(model,X_test,y_test)
    print("This is the accuracy of the model:",accuracy)

if __name__ == "__main__":
    main()


