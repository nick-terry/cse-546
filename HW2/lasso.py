#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:38:18 2020

@author: nick
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def coordDescent(X,y,lambda_reg,w=None,b=None,delta=1e-3):
    
    # If no w,b are specified, initialize w as ones vector
    if w is None:
        w = np.zeros(shape=(X.shape[1],1))
    
    # Loop until converged
    deltaW = np.inf*np.ones(shape=(X.shape[1],1))
    while np.any(deltaW >= delta):
        
        # Compute b from w
        b = np.mean(y - np.expand_dims(np.sum(X @ w,axis=1),axis=1))
    
        # Compute a vector
        a = np.expand_dims(2*np.sum(X**2,axis=0),1)
        
        
        c = np.zeros((X.shape[1],1))
        oldW = deepcopy(w)
        
        for k in range(X.shape[1]):
            w[k] = 0
            
            # Compute c vector entry
            c[k] = 2*np.sum(np.expand_dims(X[:,k],axis=1)*(y-(b + X @ w)))
                
            # Compute new w vector entry
            if c[k] < -lambda_reg:
                w[k] = (c[k]+lambda_reg)/a[k]
            elif c[k] > lambda_reg:
                w[k] = (c[k]-lambda_reg)/a[k]
            # Otherwise, we leave it as zero
        
        # Compute change in w
        deltaW = w - oldW
        
        # Compute the objective function value
        objFn = np.sum((X @ w + b - y)**2) + lambda_reg*np.linalg.norm(w,1)
        
    return w,b

def generateData(n,d,k,sigma):
    
    # Create w vector
    w = np.expand_dims(np.append(np.array(range(1,k+1))/k,np.zeros((d-k))),
                       axis=0)
    
    # Randomly generate x and epsilon vectors
    X = np.random.normal(0,1,size=(n,d))
    epsilon = np.expand_dims(np.random.normal(0,sigma,size=n),axis=-1)
    
    # Compute y
    y = X @ w.T + epsilon
    
    return X,y,w.T

def getLambdaMax(X,y):
    return np.max(2*np.abs(np.sum(X*(y-np.mean(y)),axis=0)))

def solveLASSO(X,y,wTrue=None,lambdaDecay=.66):
    
    # Solve the LASSO with given regularization path
    w = np.zeros((X.shape[1],1))
    
    # If no lambda is specified, compute lambdaMax
    lambda_reg = getLambdaMax(X,y)
    
    lambdaList = []
    nonZeroEntriesList = []
    fdrList = []
    tprList = []
    
    while lambda_reg > 1e-3:
        
        lambdaList.append(lambda_reg)
        
        # Solve LASSO w/ coordinate descent
        w,b = coordDescent(X, y, lambda_reg=lambda_reg)
        print('# selected variables: {}'.format(np.sum(w>0)))
        
        # Record number of non-zero entries in w
        numNonzero = np.sum(w>0)
        nonZeroEntriesList.append(numNonzero)
        
        if wTrue is not None:
            # Record false discovery rate
            fdrList.append(np.sum((w!=0) * (wTrue==0))/numNonzero)
            
            # Record true positive rate
            tprList.append(np.sum((w!=0) * (wTrue!=0))/np.sum(wTrue!=0))
        
        # Decrease lambda_reg
        lambda_reg = lambdaDecay * lambda_reg
        
        print('Trying new lambda={}'.format(lambda_reg))
    
    return lambdaList,nonZeroEntriesList,fdrList,tprList

def syntheticDataLASSO():
    
    # Generate some data
    d = 1000
    k = 100
    X,y,wTrue = generateData(n=500, d=d, k=k, sigma=1)
    
    return solveLASSO(X, y, wTrue)

if __name__=='__main__':
    
    np.random.seed(420)
    lambdaList,nonZeroEntriesList,fdrList,tprList = syntheticDataLASSO()
    
    # Plot 1
    fig,ax = plt.subplots(1)
    ax.plot(lambdaList,nonZeroEntriesList)
    ax.set_xscale('log')
    ax.set_xlabel('lambda_reg')
    ax.set_ylabel('# of non-zero entries in w')
    
    # Plot 2
    fig,ax = plt.subplots(1)
    ax.plot(fdrList,tprList)
    ax.set_xlabel('False Discovery Rate')
    ax.set_ylabel('True Positive Rate')