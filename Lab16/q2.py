# Implement XGBoost classifier and regressor using scikit-learn

#### XGBoost Regressor
### Boston
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def XGBoost_Regressor_implementation():
    def load_data():
        df = pd.read_csv('/home/ibab/ML_data/Boston.csv')
        X = df.drop(columns=['Unnamed: 0', 'medv'], axis=1)
        y = df['medv']
        return X, y

    X, y = load_data()

    def preprocessing(X):
        num_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'lstat']
        cat_features = ['chas', 'rad']
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ('simpleImputer', SimpleImputer(strategy='most_frequent')),
            ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features),
        ])
        x_scaled = preprocessor.fit_transform(X)
        return x_scaled

    x_scaled = preprocessing(X)

    def split_data(x_scaled):
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(x_scaled)

    def model():
        model = XGBRegressor(n_estimators=100, learning_rate=1.0, random_state=90)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    print("The R2 score for XGBRegressor on Boston dataset is ",model())


### XGBoost Classifier for Weekly dataset

def XGBoost_Classifier_Implementation():
    def load_data():
        df = pd.read_csv('/home/ibab/ML_data/Weekly.csv')
        X = df.drop(columns=['Direction'], axis=1)
        y = df['Direction']
        Y = y.map({'Down': 0, 'Up': 1})
        return X, Y

    X, Y = load_data()

    def preprocessing(X):
        num_features = ['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
        ])
        x_scaled = preprocessor.fit_transform(X)
        return x_scaled

    x_scaled = preprocessing(X)

    def split_data(x_scaled):
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(x_scaled)

    def model():
        model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    print("The accuracy of XGBClassifier for weekly dataset is ",model())


def main():
    XGBoost_Regressor_implementation()
    XGBoost_Classifier_Implementation()

if __name__ == '__main__':
    main()