# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:24:02 2017

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def get_data():
    df=pd.read_csv("voice.csv")
    X=df.iloc[:, :-1]
    y=df.iloc[:,-1]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # male -> 1
    # female -> 0
    print y
    print X
    gender_encoder = LabelEncoder()
    y = gender_encoder.fit_transform(y) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print y_train
    return X_train, X_test, y_train, y_test
    
def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') #Default hyperparameters
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test,y_pred))
    

X_train, X_test, y_train, y_test=get_data()
predict_svm(X_train, X_test, y_train, y_test)