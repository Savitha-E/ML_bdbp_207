# For the diabetes.csv dataset, build a logistic regression classifier to predict the risk of heart disease.
# Vary the threshold to generate multiple confusion matrices.  Implement a python code to calculate the following metrics
# Accuracy
# Precision
# Sensitivity
# Specificity
# F1-score
# Plot the ROC curve
# AUC
from statistics import LinearRegression

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,auc,roc_curve,roc_auc_score,classification_report,confusion_matrix
import pandas as pd
from sklearn.impute import SimpleImputer

from Git_bdbp_207.Lab19.q1 import X_train


def load_data():
    df=pd.read_csv('/home/ibab/ML_data/diabetes.csv')
    X=df.drop('Outcome',axis=1)
    y=df['Outcome']
    return X,y

X,y=load_data()
# print(load_data())

def preprocessing(X):
    num=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

    num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                 ('scaler',StandardScaler())])
    preprocessor=ColumnTransformer(transformers=[('num',num_pipeline,num)])
    x_prep=preprocessor.fit_transform(X)
    return x_prep

x_preprocessed=preprocessing(X)
# print(x_preprocessed)

def split_data(x_preprocessed,y):
    X_train,X_test,y_train,y_test = train_test_split(x_preprocessed,y,test_size=0.2)
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=train_test_split(x_preprocessed,y,test_size=0.2)



def model():
     model=LogisticRegression(random_state=123)
     model.fit(X_train,y_train)
     y_pred=model.predict(X_test)
     y_prob=model.predict_proba(X_test)[:,1]
     return y_pred,y_prob

y_pred,y_prob=model()
# print(model())

def evaluation_metrics(y_pred,y_test):
  thresholds=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  for t in thresholds:
      print("For t=",t)
      cm=confusion_matrix(y_test,y_pred)
      TN,FN ,FP , TP=cm.ravel()

      accuracy=(TN+TP)/(TN+TP+FN+FP)
      precision=(TP+TP)/(TP+FP+TN+FP)
      recall=(TP+TP)/(TP+FN+TN+FP)
      specificity=(TN+TP)/(TN+FP+TN+FN)
      f1=(2*precision*recall)/(precision+recall)
      print('Accuracy:',accuracy)
      print('Precision:',precision)
      print('Recall:',recall)
      print('Specificity:',specificity)
      print('F1:',f1)
      print("===============================================")

print(evaluation_metrics(y_pred,y_test))


def plot_roc_curve(y_pred,y_test):
    fpr, tpr, thresholds = roc_curve(y_test,y_pred)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

print(plot_roc_curve(y_pred,y_test))

