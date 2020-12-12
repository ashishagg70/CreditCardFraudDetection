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

from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('creditcard.csv')

df['samount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

data = df.drop(['Time','Amount'],axis=1)

X=df.drop(columns=['Class'])
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

rnd_clf=RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)

print('Before any balancing.')

print('---'*10)

print("precision={},recall={},f1_score={}".format(precision_score(y_test,y_pred),recall_score(y_test,y_pred),f1_score(y_test,y_pred)))

print("accuracy={}".format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))

X = pd.concat([X_train, y_train], axis=1)

valid = X[X['Class']==0]
fraud = X[X['Class']==1]

valid_down = resample(valid,replace=True,n_samples=len(fraud))

downsampled_data = pd.concat([fraud, valid_down])

y_train = downsampled_data['Class']
X_train = downsampled_data.drop('Class', axis=1)

rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)
print('After Down Sampling.')

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

rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)

print('---'*10)

print("precision={},recall={},f1_score={}".format(precision_score(y_test,y_pred),recall_score(y_test,y_pred),f1_score(y_test,y_pred)))

print("accuracy={}".format(accuracy_score(y_test,y_pred)))

print(confusion_matrix(y_test,y_pred))


