# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:24:02 2017

@author: lenovo
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_data():
    df=pd.read_csv("voice.csv")
    X=df.iloc[:, :-1]
    y=df.iloc[:,-1]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # male -> 1
    # female -> 0
    gender_encoder = LabelEncoder()
    y = gender_encoder.fit_transform(y) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
    
def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') #Default hyperparameters
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score with svm:')
    print(metrics.accuracy_score(y_test,y_pred))
    
def predict_mlp(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with mlp:')
    print(metrics.accuracy_score(y_test,y_pred))    
    
def predict_rf(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with random forest:')
    print(metrics.accuracy_score(y_test,y_pred)) 
    
def predict_dt(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with decision trees:')
    print(metrics.accuracy_score(y_test,y_pred)) 

def predict_et(X_train, X_test, y_train, y_test):
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with extra trees:')
    print(metrics.accuracy_score(y_test,y_pred))

def predict_ada(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with AdaBoost:')
    print(metrics.accuracy_score(y_test,y_pred))

def predict_gb(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)  
    print('Accuracy Score with Gradient Boosting:')
    print(metrics.accuracy_score(y_test,y_pred))

X_train, X_test, y_train, y_test=get_data()
predict_gb(X_train, X_test, y_train, y_test)