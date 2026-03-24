# Implement Adaboost classifier using scikit-learn. Use the Iris dataset.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def load_data():
    df=pd.read_csv('/home/ibab/ML_data/iris.csv')
    X=df.drop(['species'],axis=1)
    y=df['species']
    Y=y.map({'setosa':0,'versicolor':1,'virginica':2})
    return X,y

X,y=load_data()


def preprocessing(X):
    num_features=['sepal_length','sepal_width','petal_length','petal_width']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    preprocessor=ColumnTransformer([
        ('num', num_pipeline, num_features),
    ])
    x_scaled=preprocessor.fit_transform(X)
    return x_scaled

x_scaled=preprocessing(X)

def split_data(x_scaled):
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=split_data(x_scaled)

def model():
    model=AdaBoostClassifier( estimator=DecisionTreeClassifier(),n_estimators=100, learning_rate=1.0 , random_state=90)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy

print(model())
