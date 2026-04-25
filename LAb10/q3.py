import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------
# Load sonar data
# -------------------------------
def load_data():
    df = pd.read_csv("sonar.csv", header=None)
    return df


# -------------------------------
# Split features and target
# -------------------------------
def split_features_target(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


# -------------------------------
# Train classification tree
# -------------------------------
def train_classification_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    model.fit(X_train, y_train)
    return model


# -------------------------------
# Evaluate model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# -------------------------------
# Main program
# -------------------------------
df = load_data()

print("First 5 rows:")
print(df.head())

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = train_classification_tree(X_train, y_train)

evaluate_model(model, X_test, y_test)