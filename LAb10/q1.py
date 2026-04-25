import pandas as pd
import numpy as np


# -------------------------------
# Create simulated data
# -------------------------------
def create_data():
    np.random.seed(0)

    data = {
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "BP":  [72, 78, 79, 80, 81, 82, 85, 88, 90, 95],
        "Disease_Score": [20, 25, 30, 35, 40, 42, 50, 55, 60, 70]
    }

    df = pd.DataFrame(data)
    return df


# -------------------------------
# Partition data
# -------------------------------
def partition_data(df, threshold):
    left_data = df[df["BP"] <= threshold]
    right_data = df[df["BP"] > threshold]

    return left_data, right_data


# -------------------------------
# Main program
# -------------------------------
df = create_data()

print("Original Dataset:")
print(df)

thresholds = [80, 78, 82]

for t in thresholds:
    left, right = partition_data(df, t)

    print("\n================================")
    print("Threshold:", t)
    print("Left partition: BP <=", t)
    print(left)

    print("\nRight partition: BP >", t)
    print(right)