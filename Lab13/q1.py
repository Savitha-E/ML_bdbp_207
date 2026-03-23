#Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor,BaggingClassifier
import pandas as pd

def question1():
    print("Question 1, Bagging classifier for Daibetes")

    def load_data():
        df = pd.read_csv('/home/ibab/ML_data/diabetes.csv')
        X = df.drop(columns=['Outcome'], axis=1)
        Y = df['Outcome']
        return X, Y

    X, Y = load_data()

    def preprocessing(X):
        '''
         #No categorical features, so One hot encoding is not required and the target also has already been Label encoded
        # as 1 and 0.
        '''
        num_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age']
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))])
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
        ])
        X_scaled = preprocessor.fit_transform(X)
        return X_scaled

    x_scaled = preprocessing(X)

    def split_data(x_scaled, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(x_scaled, Y)

    def model(x_train, x_test, y_train, y_test):
        bagging_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
        bagging_classifier.fit(x_train, y_train)
        y_pred = bagging_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    print("The accuracy of the bagging classifier is=",model(x_train, x_test, y_train, y_test))


def question2():
    print("Question 2, Bagging classifier for iris.csv")
    def load_data():
        df = pd.read_csv('/home/ibab/ML_data/iris.csv')
        X = df.drop(columns=['species'], axis=1)
        y = df['species']
        Y = y.map({"setosa": 1, "versicolor": 2, "virginica": 3})
        return X, Y

    X, Y = load_data()

    # print(X)

    def preprocessing(X):
        num_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
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

    # print(preprocessing(X))

    def split_data(x_scaled, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split_data(x_scaled, Y)

    def model(x_train, x_test, y_train, y_test):
        bagging_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
        bagging_classifier.fit(x_train, y_train)
        y_pred = bagging_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    print("The accuracy of the bagging classifier is=",model(x_train, x_test, y_train, y_test))


def question3():
    print("Question 3, Bagging regressor for simulated_dataset.csv")
    def load_data():
        df = pd.read_csv('/home/ibab/ML_data/simulated_dataset.csv')
        X = df.drop(columns=['disease_score'], axis=1)
        Y = df['disease_score']
        return X, Y

    X, Y = load_data()

    def preprocessing(X):
        num_features = ["age", "BMI", "BP", "blood_sugar", "disease_score_fluct"]
        categorical_features = ["Gender"]
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder())
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("categorical", categorical_pipeline, categorical_features),
        ])

        x_scaled = preprocessor.fit_transform(X)
        return x_scaled

    x_scaled = preprocessing(X)

    def data_split(x_scaled, Y):
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = data_split(x_scaled, Y)

    def model(x_train, x_test, y_train, y_test):
        bagging_regressor = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=100,
            random_state=42,
        )
        bagging_regressor.fit(x_train, y_train)
        y_pred = bagging_regressor.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return r2

    print("r2 of the bagging regressor model is", model(x_train, x_test, y_train, y_test))


def main():
 question1()
 print()
 question2()
 print()
 question3()

if __name__ == "__main__":
    main()
