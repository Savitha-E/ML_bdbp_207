import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('/home/ibab/ML_data/diabetes.csv')

print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())

#-------------Drop unwanted columns-------------
# df = df.drop(['Unnamed: 0'], axis=1)
# print(df.columns)

#------------Add Missing values------------------
num_cols=df.select_dtypes(include=np.number).columns
cat_cols=df.select_dtypes(include='object').columns

for col in num_cols:
    df[col]=df[col].fillna(df[col].median())
for col in cat_cols:
    df[col]=df[col].fillna(df[col].mode()[0])
print("After adding missing values:")
print(df.isnull().sum())

#-------------EDA---------------------------------
plt.figure()
sns.countplot(x='Outcome', data=df)
plt.title('Number of Diabetic patients')
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#------------Removing Outliers---------------------

