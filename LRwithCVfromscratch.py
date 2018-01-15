# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:06:11 2017

@author: Likitha
"""

import pandas as pd
import numpy as np

#extracting data
df = pd.read_csv("irisdata/iris-dataset.txt",names=["Sepal length","sepal width","petal length","petal width","class"],header=None)
df = df.replace({'class' : { 'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3 }})
df = df.as_matrix()

#setting variables
array = np.insert(df,0,1,axis=1)
array = np.delete(array, -1,axis=1)
X = array
Y= df[:,-1]

#Splitting of data for k-fold cross validation and then calculating average accuracy
def kfoldcv(folds):
    accuracy_list = []
    fold = int(len(X)/folds)
    X_split = np.split(X,folds)
    Y_split = np.split(Y,folds)
    for i in range(folds):
        X_test = np.array(X_split[i])
        Y_test = np.array(Y_split[i])
        X_train = np.delete(X_split,i,0)
        X_train = np.concatenate(X_train)
        Y_train = np.delete(Y_split,i,axis=0)
        Y_train = np.concatenate(Y_train)
        exp_Y = np.dot(X_test,beta(X_train,Y_train))
        count = 0
        for item in np.equal(Y_test,exp_Y.round()):
            if item == True:
                count+=1
        accuracy = count/fold
        accuracy_list.append(accuracy)
    return ((sum(accuracy_list)/len(accuracy_list))*100) 

#calculate B hat
def beta(X,Y):
    X_tr = np.transpose(X)
    mul = np.dot((X_tr),X)
    inv = np.linalg.inv(mul)
    temp = np.dot((inv),(X_tr))
    B_hat = np.dot(temp,Y)
    return B_hat

#Cross Validation of data
print ("Accuracy for 2 fold cross validation is: {}".format(kfoldcv(2)))
print ("Accuracy for 5 fold cross validation is: {}".format(kfoldcv(5)))
print ("Accuracy for 10 fold cross validation is: {}".format(kfoldcv(10)))
print ("Accuracy for leave one out cross validation is: {}".format(kfoldcv(150)))