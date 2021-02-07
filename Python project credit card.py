# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:26:18 2021

@author: Sravanthi TC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv("C:/Users/Sravanthi TC/Downloads/Part 2/Part 2/Class 14/Assignment/Case 2/creditcard.csv")
print(data.head())
print(data.shape)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

model=LogisticRegression(random_state=10).fit(train_x,train_y)
pred_y=model.predict(test_x)

print(accuracy_score(test_y,pred_y))
print(confusion_matrix(test_y,pred_y))
print(classification_report(test_y,pred_y))

# The accuracy score shows that the data is unbalanced

sns.countplot(data.Class,label="class",color="red")
plt.show()

# Target having value 0 is very large compared to target having 1.

target0=data[data.Class==0]
print(len(target0))
target1=data[data.Class==1]
print(len(target1))

balanced_data=pd.concat([target1,target0.sample(n=len(target1),random_state=10)])

sns.countplot(balanced_data.Class,label="class",color="blue")
plt.show()

X=balanced_data.iloc[:,:-1]
Y=balanced_data.iloc[:,-1]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.25,random_state=10)

# logistic regression

logit_model=LogisticRegression(random_state=10).fit(train_x,train_y)
pred_y_logit=logit_model.predict(test_x)

logit_model_score=accuracy_score(test_y,pred_y_logit)
print(logit_model_score)
print(confusion_matrix(test_y,pred_y_logit))
print(classification_report(test_y,pred_y_logit))

# decision tree classifier-gini method

dt_gini_model=DecisionTreeClassifier(criterion="gini",random_state=10).fit(train_x,train_y)
pred_y_dt_gini=dt_gini_model.predict(test_x)  

dt_gini_model_score=accuracy_score(test_y,pred_y_dt_gini)
print(dt_gini_model_score)
print(confusion_matrix(test_y,pred_y_dt_gini))
print(classification_report(test_y,pred_y_dt_gini))

# decision tree classifier-entropy method

dt_entropy_model=DecisionTreeClassifier(criterion="entropy",random_state=10).fit(train_x,train_y)
pred_y_dt_entropy=dt_entropy_model.predict(test_x)  

dt_entropy_model_score=accuracy_score(test_y,pred_y_dt_entropy)
print(dt_entropy_model_score)
print(confusion_matrix(test_y,pred_y_dt_entropy))
print(classification_report(test_y,pred_y_dt_entropy))

# random forest classifier -gini method

rf_gini_model=RandomForestClassifier(criterion="gini",random_state=10).fit(train_x,train_y)
pred_y_rf_gini=rf_gini_model.predict(test_x)  

rf_gini_model_score=accuracy_score(test_y,pred_y_rf_gini)
print(rf_gini_model_score)
print(confusion_matrix(test_y,pred_y_rf_gini))
print(classification_report(test_y,pred_y_rf_gini))

# random forest classifier -entropy method

rf_entropy_model=RandomForestClassifier(criterion="entropy",random_state=10).fit(train_x,train_y)
pred_y_rf_entropy=rf_entropy_model.predict(test_x)  

rf_entropy_model_score=accuracy_score(test_y,pred_y_rf_entropy)
print(rf_entropy_model_score)
print(confusion_matrix(test_y,pred_y_rf_entropy))
print(classification_report(test_y,pred_y_rf_entropy))

# Gradient Boosting clasifier

gbm_model=GradientBoostingClassifier(random_state=10).fit(train_x,train_y)
pred_y_gbm=gbm_model.predict(test_x)  

gbm_model_score=accuracy_score(test_y,pred_y_gbm)
print(gbm_model_score)
print(confusion_matrix(test_y,pred_y_gbm))
print(classification_report(test_y,pred_y_gbm))

# SVM

svm_model=GridSearchCV(svm.SVC(random_state=10),{'C':[1,5,10],'kernel':['rbf','linear'],'decision_function_shape':['ovr','ovo']},cv=5).fit(train_x,train_y)
pred_y_svm=svm_model.predict(test_x)  

svm_model_score=accuracy_score(test_y,pred_y_svm)
print(svm_model_score)
print(confusion_matrix(test_y,pred_y_svm))
print(classification_report(test_y,pred_y_svm))

svm_model.best_params_


result=pd.DataFrame({'model':['Logistic regression','decisiontreeclassifier-gini'
                             ,'decisiontreeclassifier-entropy','randomforestclassifier-gini',
                             'randomforestclassifier-entropy','gradientboostingclassifier','SVM'],
                    'score':[logit_model_score,dt_gini_model_score,dt_entropy_model_score,rf_gini_model_score,
                             rf_entropy_model_score,gbm_model_score,svm_model_score]})

print(result.sort_values(by='score',ascending=False))

 # by seeing the results, we can conclude that SVM models gives more 
 # accuracy compared to other algorithms.
            
 