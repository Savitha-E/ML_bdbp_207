# Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd

def load_data():
    df=pd.read_csv('/home/ibab/ML_data/sonar.csv')
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return X,y

X,y=load_data()

#The data is already preprocessed
def preprocessing(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled
x_scaled=preprocessing(X)

def split_data(x_scaled,y):
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=split_data(X,y)

# print(x_train,x_test,y_train,y_test)

def model():
    model=BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators=100)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy

print(model())
