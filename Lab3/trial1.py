# Use the above simulated CSV file and implement the following from scratch in Python
# Read simulated data csv file
# Form x and y (disease_score_fluct)
# Write a function to compute hypothesis
# Write a function to compute the cost
# Write a function to compute the derivative
# Write update parameters logic in the main function

import numpy as np
import pandas as pd

#1Read simulated data csv file,Form x and y (disease_score_fluct)
def load_data():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop("disease_score_fluct","disease_score", axis=1)
    y = df["disease_score_fluct"]
    print(X.shape)
    print(y.shape)
    return X,y


print(load_data()) # this will load the data
def main():
    X, y = load_data()

if __name__ == '__main__':
    main()
# Write a function to compute hypothesis
