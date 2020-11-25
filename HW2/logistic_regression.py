#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:42:24 2020

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def mu(X,y,w,b):
    return 1/(1+np.exp(-y*(b + X @ w)))

def gradW(X,y,lambda_reg,w,b):
    
    matrixExpr = (mu(X,y,w,b)-1)*y*X
    
    return np.expand_dims(np.sum(matrixExpr,axis=0),1) + 2*lambda_reg * w 

def gradb(X,y,w,b):
    
    matrixExpr = (mu(X,y,w,b)-1)*y
    
    return np.sum(matrixExpr)

def getObjFn(X,y,lambda_reg):
    
    def objFn(w,b):        
        return np.mean(np.log(1+np.exp(-y*(b + X @ w))))+lambda_reg*np.linalg.norm(w)**2
    
    return objFn

def gradDescent(X,y,lambda_reg=.3,eta=.001,eps=3e-2):
    
    # Initial w is zeros
    w = np.zeros((X.shape[1],1))
    b = 0
    
    # Get obj function
    objFn = getObjFn(X, y, lambda_reg)

    oldObj = objFn(w,b)

    # Store w and b from each iteration
    wList = [w,]
    bList = [b,]
    jList = [oldObj,]
    deltaList = []
    
    deltaObj = np.inf
    
    while len(deltaList) < 20 or deltaObj <= -eps:
    
        dW,dB = gradW(X,y,lambda_reg,w,b),gradb(X,y,w,b)
        _w,_b = w-eta*dW,b-eta*dB
        

        newObj = objFn(_w,_b)
        deltaObj = newObj-oldObj
        print('Decrease in obj fn: {}'.format(deltaObj))
        
        w = _w
        b = _b
        oldObj = newObj
        
        wList.append(w)
        bList.append(b)
        jList.append(newObj)
        deltaList.append(deltaObj)
    
    return wList,bList,jList

def stochasticGradDescent(X,y,batchSize,lambda_reg=.3,eta=.001,eps=1e-3):
    
    # Initial w is zeros
    w = np.zeros((X.shape[1],1))
    b = 0
    
    # Rescale regularization factor for the batch size
    lambda_reg = lambda_reg * batchSize / X.shape[0]
    
    # Get obj function
    objFn = getObjFn(X, y, lambda_reg)

    oldObj = objFn(w,b)

    # Store w and b from each iteration
    wList = [w,]
    bList = [b,]
    jList = [oldObj,]
    deltaList = []
    
    deltaObj = np.inf
    
    while len(deltaList) < 20 or np.mean(deltaList[-10:]) <= -eps:
        
        # Randomly select batchSize observations to compute gradient
        gradInd = np.random.choice(range(X.shape[0]),size=batchSize,replace=False)
        
        Xgrad,ygrad = X[gradInd,:],y[gradInd]
        
        dW,dB = gradW(Xgrad,ygrad,lambda_reg,w,b),gradb(Xgrad,ygrad,w,b)
        _w,_b = w-eta*dW,b-eta*dB
        
        newObj = objFn(_w,_b)
        deltaObj = newObj-oldObj
        print('Decrease in obj fn: {}'.format(deltaObj))
        
        w = _w
        b = _b
        oldObj = newObj
        
        wList.append(w)
        bList.append(b)
        jList.append(newObj)
        deltaList.append(deltaObj)
    
    return wList,bList,jList

if __name__=='__main__':
    
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    trainInd = (labels_train==2)+(labels_train==7)
    labels_train = labels_train[trainInd]
    y_train = 1*(labels_train==7)-1*(labels_train==2)
    y_train = np.expand_dims(y_train,1)
    
    testInd = (labels_test==2)+(labels_test==7)
    labels_test = labels_test[testInd]
    y_test = 1*(labels_test==7)-1*(labels_test==2)
    y_test = np.expand_dims(y_test,1)
    
    def  runGradDescent(epsilon):
        wList,bList,jListTrain = gradDescent(X_train[trainInd,:],y_train,eps=epsilon)
        
        # Compute jListTest
        testObj = getObjFn(X_test[testInd,:], y_test, .1)
        jListTest = []
        for i in range(1,len(wList)):
            jListTest.append(testObj(wList[i],bList[i]))
        
        fig,ax = plt.subplots(1)
        ax.plot(jListTrain[1:],alpha=.5,label='Train Objective Function Value')
        ax.plot(jListTest,alpha=.5,label='Test Objective Function Value')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('J(w,b)')
        ax.legend()
        
        # Compute the classification errors
        
        classErrTrain = []
        classErrTest = []
        for i in range(1,len(wList)):
            train_preds = np.sign(bList[i] + X_train[trainInd,:] @ wList[i])
            test_preds = np.sign(bList[i] + X_test[testInd,:] @ wList[i])
            classErrTrain.append(np.sum(train_preds!=y_train)/y_train.size*100)
            classErrTest.append(np.sum(test_preds!=y_test)/y_test.size*100)
            
        fig,ax = plt.subplots(1)
        ax.plot(classErrTrain,label='Train Classification Error')
        ax.plot(classErrTest,label='Test Classification Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Classification Error (%)')
        ax.legend()
        
    def  runStochasticGradDescent(batchSize,epsilon):
        wList,bList,jListTrain = stochasticGradDescent(X_train[trainInd,:],y_train,batchSize=batchSize,
                                                       eps=epsilon)
        
        # Compute jListTest
        testObj = getObjFn(X_test[testInd,:], y_test, .1)
        jListTest = []
        for i in range(1,len(wList)):
            jListTest.append(testObj(wList[i],bList[i]))
        
        fig,ax = plt.subplots(1)
        ax.plot(jListTrain[1:],label='Train Objective Function Value')
        ax.plot(jListTest,label='Test Objective Function Value')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('J(w,b)')
        ax.legend()
        
        # Compute the classification errors
        
        classErrTrain = []
        classErrTest = []
        for i in range(1,len(wList)):
            train_preds = np.sign(bList[i] + X_train[trainInd,:] @ wList[i])
            test_preds = np.sign(bList[i] + X_test[testInd,:] @ wList[i])
            classErrTrain.append(np.sum(train_preds!=y_train)/y_train.size*100)
            classErrTest.append(np.sum(test_preds!=y_test)/y_test.size*100)
            
        fig,ax = plt.subplots(1)
        ax.plot(classErrTrain,label='Train Classification Error')
        ax.plot(classErrTest,label='Test Classification Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Classification Error (%)')
        ax.legend()

if __name__=='__main__':
    
    runGradDescent(epsilon=.03)
    runStochasticGradDescent(batchSize=1, epsilon=1e-5)
    runStochasticGradDescent(batchSize=100, epsilon=1e-5)