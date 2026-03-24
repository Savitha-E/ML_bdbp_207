#Implement Gradient Boost Regression and Classification using scikit-learn.
# Use the Boston housing dataset from the ISLP package for the regression problem and
# weekly dataset from the ISLP package and use Direction as the target variable for the classification.


### Boston
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def load_data():
    df=pd.read_csv('/home/ibab/ML_data/Boston.csv')
    X=df.drop(columns=['Unnamed: 0','medv'],axis=1)
    y=df['medv']
    return X,y

X,y=load_data()

def preprocessing(X):
    num_features=['crim','zn','indus','nox','rm','age','dis','tax','ptratio','lstat']
    cat_features=['chas','rad']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('simpleImputer', SimpleImputer(strategy='most_frequent')),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor=ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
    ])
    x_scaled=preprocessor.fit_transform(X)
    return x_scaled

x_scaled=preprocessing(X)

def split_data(x_scaled):
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=split_data(x_scaled)

def model():
    model=GradientBoostingRegressor(n_estimators=100, learning_rate=1.0 , random_state=90)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    r2=r2_score(y_test,y_pred)
    return r2

print(model())
