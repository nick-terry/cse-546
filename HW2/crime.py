#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:58:09 2020

@author: nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasso

df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")

# Get column names of response, predictors
columns = list(df_train.columns)
resp,preds = columns[0],columns[1:]

# Get train and test datasets
y_train,X_train = np.expand_dims(df_train[resp].to_numpy(),1),df_train[preds].to_numpy()
y_test,X_test = np.expand_dims(df_test[resp].to_numpy(),1),df_test[preds].to_numpy()

resultList = []

# Run LASSO with lambda_max first
lambda_reg = lasso.getLambdaMax(X_train,y_train)

# Repeatedly solve LASSO by decreasing lambda_reg by factor of 2
while lambda_reg >= .01:
    
    w,b = lasso.coordDescent(X_train,y_train,lambda_reg)    
    resultList.append((w,b,lambda_reg))
    
    # Reduce lambda_reg by factor of 2
    lambda_reg = .5 * lambda_reg


lambdaList = [item[2] for item in resultList]
wList,bList = [item[0] for item in resultList],[item[1] for item in resultList]

# Create plot 1 of lambda versus number of non-zero elements in w
fig,ax = plt.subplots(1)
ax.plot(lambdaList,[np.sum(w!=0) for w in wList])
ax.set_xscale('log')
ax.set_xlabel('lambda_reg')
ax.set_ylabel('# of non-zero entries in w')

# Create plot 2 of regularization paths for some predictors
predNames = ['agePct12t29','pctWSocSec','pctUrban','agePct65up','householdsize']
predIndices = [columns.index(name) for name in predNames]
predCoeffsDict = {}

for i,name in enumerate(predNames):
    predCoeffsDict[name] = [w[predIndices[i]] for w in wList]

fig,ax = plt.subplots(1)
for name in predCoeffsDict:
    ax.plot(lambdaList,predCoeffsDict[name],label=name)
ax.set_xscale('log')
ax.set_xlabel('lambda_reg')
ax.set_ylabel('w coefficient')
ax.legend()

# Create plot 3 of lambda_reg versus squared errors
trainErr,testErr = [],[]
for w,b in zip(wList,bList):
    trainErr.append(np.mean((X_train @ w + b - y_train)**2))
    testErr.append(np.mean((X_test @ w + b - y_test)**2))
    
fig,ax = plt.subplots(1)
ax.plot(lambdaList,trainErr,label='Train')
ax.plot(lambdaList,testErr,label='Test')
ax.set_xscale('log')
ax.set_xlabel('lambda_reg')
ax.set_ylabel('mean squared error')
ax.legend()

# Part d
# Solve the LASSO with lambda_reg=30
w,b = lasso.coordDescent(X_train,y_train,30)

# Get feature with largest coefficient
maxInd = np.argmax(w)
print('Largest coefficient: {}'.format(columns[maxInd])) # NumIlleg

# Get feature with most negative coefficient
minInd = np.argmin(w)
print('Smallest coefficient: {}'.format(columns[minInd])) # PctFam2Par