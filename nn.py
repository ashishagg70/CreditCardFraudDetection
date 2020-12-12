import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score,classification_report,roc_curve,auc

from sklearn.neural_network import MLPClassifier

df=pd.read_csv('creditcard.csv')

df['samount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

data = df.drop(['Time','Amount'],axis=1)

X=df.drop(columns=['Class'])
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

mlp_clf=MLPClassifier(hidden_layer_sizes=(100),max_iter=100)

mlp_clf.fit(X_train,y_train)

y_pred=mlp_clf.predict(X_test)

print('Before any balancing.')

print('---'*10)

print("precision={},recall={},f1_score={}".format(precision_score(y_test,y_pred),recall_score(y_test,y_pred),f1_score(y_test,y_pred)))

print("accuracy={}".format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))

print('After Over sampling using Imblearn')
X=df.drop(columns=['Class'])
y=df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

sm = SMOTE(random_state=42,sampling_strategy=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

print('data ready')

mlp_clf.fit(X_train,y_train)
y_pred=mlp_clf.predict(X_test)

print('---'*10)

print("precision={},recall={},f1_score={}".format(precision_score(y_test,y_pred),recall_score(y_test,y_pred),f1_score(y_test,y_pred)))

print("accuracy={}".format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))