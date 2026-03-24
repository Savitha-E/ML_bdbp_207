# Write a program to partition a dataset (simulated data for regression)  into two parts, based on a feature (BP)
# and for a threshold, t = 80. Generate additional two partitioned datasets based on different threshold values
# of t = [78, 82].

import pandas as pd

def load_data():
    df = pd.read_csv('/home/ibab/ML_data/simulated_dataset.csv')
    X=df.drop(columns=['disease_score'])
    y=df['disease_score']
    return X,y
print(load_data())
