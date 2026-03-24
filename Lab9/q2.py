# Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd

def load_data():
  df =pd.read_csv('/home/ibab/ML_data/simulated_dataset.csv')
  X=df.drop(columns=['disease_score'])
  y=df['disease_score']
  return X,y

X,Y=load_data()

def preprocessing(X):
    num_features=['age','BMI','BP','blood_sugar','disease_score_fluct']
    categorical_features=['Gender']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder()),
    ])

    preprocessor=ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, categorical_features),
    ])

    x_scaled=preprocessor.fit_transform(X)
    return x_scaled

x_scaled=preprocessing(X)

def split_data(x_scaled,y):
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test=split_data(x_scaled,Y)

def model():
    model= BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    r2=r2_score(y_test,y_pred)
    return r2

print(model())