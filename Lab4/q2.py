from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

learning_rate = 0.00001
iterations = 300

def theta_column(X):
    theta = []
    for i in range(len(X[0])):
        theta.append([0])
    return theta

def hypothesis_gen(sample, theta):
    hypothesis_func = 0
    for i in range(len(theta)):
        hypothesis_func += theta[i][0] * sample[i]
    return hypothesis_func

def cost_function(theta, X, y):
    summation = 0
    m = len(X)
    for i in range(len(X)):
        hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
        summation += (hypothesis_func_for_all_samp - y[i][0]) ** 2
    cost_func = (1 / (2 * m)) * summation
    return cost_func

def make_updated_theta(theta):
    updated_theta = []
    for i in range(len(theta)):
        updated_theta.append([theta[i][0]])
    return updated_theta

def gradient_function(updated_theta, theta, X, y):
    m = len(X)
    for j in range(len(theta)):
        summation1 = 0
        for i in range(len(X)):
            hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
            summation1 += (hypothesis_func_for_all_samp - y[i][0]) * X[i][j]
        updated_theta[j][0] -= learning_rate * (1 / m) * summation1
    return updated_theta

def identify_convergence(theta, X, y, iterations):
    for i in range(iterations):
        updated_theta = make_updated_theta(theta)
        theta = gradient_function(updated_theta, theta, X, y)
    return theta

def predict(X, theta):
    predictions = []
    for i in range(len(X)):
        predictions.append(hypothesis_gen(X[i], theta))
    return predictions

def mean_squared_error_custom(y_true, y_pred):
    summation = 0
    m = len(y_true)
    for i in range(m):
        summation += (y_true[i][0] - y_pred[i]) ** 2
    return summation / m

def r2_score_custom(y_true, y_pred):
    mean_y = 0
    m = len(y_true)
    for i in range(m):
        mean_y += y_true[i][0]
    mean_y /= m
    ss_total = 0
    ss_residual = 0
    for i in range(m):
        ss_total += (y_true[i][0] - mean_y) ** 2
        ss_residual += (y_true[i][0] - y_pred[i]) ** 2
    return 1 - (ss_residual / ss_total)

print("For California Housing Dataset using my linear model ")

california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = [[val] for val in y_train.tolist()]
y_test = [[val] for val in y_test.tolist()]

theta = theta_column(X_train)
final_theta = identify_convergence(theta, X_train, y_train, iterations)

y_pred = predict(X_test, final_theta)

print("MSE:", mean_squared_error_custom(y_test, y_pred))
print("R2 Score:", r2_score_custom(y_test, y_pred))


print("For Simulated Dataset using my linear model model")

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = df.drop("disease_score", axis=1)
y = df["disease_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = [[val] for val in y_train.tolist()]
y_test = [[val] for val in y_test.tolist()]

theta = theta_column(X_train)
final_theta = identify_convergence(theta, X_train, y_train, iterations)

y_pred = predict(X_test, final_theta)

print("MSE:", mean_squared_error_custom(y_test, y_pred))
print("R2 Score:", r2_score_custom(y_test, y_pred))


print("California Housing Dataset using scikit-learn model")

california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


print("Simulated Dataset using scikit-learn model")

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = df.drop("disease_score", axis=1)
y = df["disease_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
