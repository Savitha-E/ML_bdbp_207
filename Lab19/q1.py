# For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.
# Vary the threshold to generate multiple confusion matrices.  Implement a python code to calculate the following metrics
# Accuracy
# Precision
# Sensitivity
# Specificity
# F1-score
# Plot the ROC curve
# AUC

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

def load_data():
    df=pd.read_csv("/home/ibab/ML_data/Heart.csv")
    X=df.drop(columns=['Unnamed: 0', 'AHD'], axis=1)
    Y=df['AHD']
    y=Y.map({'Yes':1,'No':0})
    return X,y

X,y=load_data()


def preprocessing(X):
    num_cols= ['Age','RestBP','Chol','MaxHR','Oldpeak']
    cat_cols= ['Sex','ChestPain','Fbs','RestECG','ExAng','Slope','Ca','Thal']
    numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])
    categorical_pipeline =Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('numerical', numerical_pipeline, num_cols),
                                      ('categorical', categorical_pipeline, cat_cols),])
    x_prep = preprocessor.fit_transform(X)
    return x_prep

x_preprocessed=preprocessing(X)
# print(x_preprocessed)

def split_data(x_preprocessed,y):
    X_train, X_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(x_preprocessed,y)
# print(X_train,X_test,y_train,y_test)

def model(X_train,X_test,y_train,y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    return y_prob , y_test

y_prob, y_test=model(X_train,X_test,y_train,y_test)
# print(y_prob,y_test)

def evaluate_model(y_prob,y_test):
    thresholds=[0.2,0.4,0.5,0.7]

    for t in thresholds:
        print("For threshold",t)
        y_pred = (y_prob>t).astype(int)
        cm = confusion_matrix(y_test,y_pred)
        print("The cm is",cm)
        tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()

        accuracy= (tp+tn)/(tn+fn+fp+tp)
        precision= tp/(tp+fp)
        recall= tp/(tp+fn)
        sensitivity= tp/(tp+fn)
        f1= 2*precision*recall/(precision+recall)
        print("The accuracy is",accuracy)
        print("The precision is",precision)
        print("The recall is",recall)
        print("The sensitivity is",sensitivity)
        print("The F1 is",f1)
        print("==========================================================================")

print(evaluate_model(y_prob,y_test))




