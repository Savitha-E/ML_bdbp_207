#Implement Gradient Boost Regression and Classification using scikit-learn.
# Use the Boston housing dataset from the ISLP package for the regression problem and
# weekly dataset from the ISLP package and use Direction as the target variable for the classification.


### Weekly

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def load_data():
    df=pd.read_csv('/home/ibab/ML_data/Weekly.csv')
    X=df.drop(columns=['Direction'],axis=1)
    y=df['Direction']
    Y = y.map({'Down': 0, 'Up': 1})
    return X,Y

X,Y=load_data()


def preprocessing(X):
    num_features=['Year','Lag1','Lag2','Lag3','Lag4','Lag5','Volume']

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
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,Y,test_size=0.2)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=split_data(x_scaled)

def model():
    model=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy

print(model())
