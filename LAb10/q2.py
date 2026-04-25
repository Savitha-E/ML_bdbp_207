import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# Create simulated regression data
# -------------------------------
def create_data():
    np.random.seed(0)

    n = 100

    age = np.random.randint(20, 80, n)
    bp = np.random.randint(60, 120, n)
    sugar = np.random.randint(70, 200, n)

    disease_score = 0.3 * age + 0.5 * bp + 0.2 * sugar + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "Age": age,
        "BP": bp,
        "Sugar": sugar,
        "Disease_Score": disease_score
    })

    return df


# -------------------------------
# Split features and target
# -------------------------------
def split_features_target(df):
    X = df[["Age", "BP", "Sugar"]]
    y = df["Disease_Score"]
    return X, y


# -------------------------------
# Train regression tree
# -------------------------------
def train_regression_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=0, max_depth=3)
    model.fit(X_train, y_train)
    return model


# -------------------------------
# Evaluate model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)


# -------------------------------
# Main program
# -------------------------------
df = create_data()

print("First 5 rows:")
print(df.head())

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = train_regression_tree(X_train, y_train)

evaluate_model(model, X_test, y_test)