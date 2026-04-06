#Implement Random Forest algorithm for regression and classification using scikit-learn. Use diabetes and iris datasets.

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
import pandas as pd


def question1():
    def load_data():
        path = "/home/ibab/ML_data/"
        df = pd.read_csv(path + "diabetes.csv")
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        return X, y

    X, y = load_data()

    def preprocessing(X):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                       'DiabetesPedigreeFunction', 'Age']
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_columns),
        ])
        x_preprocess = preprocessor.fit_transform(X)
        return x_preprocess

    x_preprocessed = preprocessing(X)

    # print(x_preprocessed)

    def split_data(x_preprocessed, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = split_data(X, y)

    def model():
        model = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    print("The accuracy of the RandomForestClassifier for Diabetes dataset is follows:", model())

def question2():
    def load_data():
        path = "/home/ibab/ML_data/"
        df = pd.read_csv(path + "simulated_dataset.csv")
        X = df.drop(columns=["disease_score"])
        y = df["disease_score"]
        return X, y

    X, y = load_data()


    def preprocessing(X):
        num_columns = ['age', 'BMI', 'BP', 'blood_sugar', 'disease_score_fluct']
        categorical_columns = ['Gender']
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_columns),
            ('cat', cat_pipeline, categorical_columns),
        ])
        x_preprocess = preprocessor.fit_transform(X)
        return x_preprocess

    x_preprocessed = preprocessing(X)

    def split_data(x_preprocessed, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = split_data(X, y)

    def model():
        model = BaggingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    print("The accuracy of the RandomForestRegressor for Simulated dataset is follows:", model())


def main():
    question1()
    question2()

if __name__ == "__main__":
    main()