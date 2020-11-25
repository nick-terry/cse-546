#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:36:16 2020

@author: nick
"""

import numpy as np
import scipy.optimize as opt
import itertools
import matplotlib.pyplot as plt

np.random.seed(42)

def generateData(n):
    
    x = np.random.uniform(0,1,size=(n,1))
    y = f(x)
    eps = np.random.normal(0,1,size=(n,1))
    return x,y,y+eps

def f(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

def RBF(x,xPrime,gamma):
    diff = x-xPrime.T
    return np.exp(-gamma*diff**2)

def Poly(x,xPrime,d):
    outer = x @ xPrime.T
    return (1+outer)**d

def K(x,xPrime,k):
    K = k(x,xPrime)
    return K

def loss(alpha,K,y,lambda_reg):
    return np.linalg.norm(K @ alpha[:,None] - y,ord=2)**2 + lambda_reg * alpha.T @ K @ alpha

def getPredictor(x,y,k,lambda_reg):
    
    Kmat = K(x,x,k)
    alpha = np.linalg.solve(Kmat+lambda_reg*np.eye(Kmat.shape[0]), y)
    
    if len(alpha.shape)==1:
        alpha = alpha[:,None]
    
    def f(xPred):
        
        K = k(xPred,x)
        return K @ alpha
    
    return f

def fitPredictorRBF(x,y,k):
    
    # Choose param space for lambda_reg, kernel hyper param
    lambda_regList = [10**-z for z in range(0,11)]

    gamma0 = 1/np.median((x-x.T)**2)
    gammaList = [gamma0 - z*gamma0/10 for z in range(1,10)] +\
                          [gamma0 + z*gamma0/10 for z in range(0,10)]
    
    paramsComb = list(itertools.product(lambda_regList,gammaList))
    scoreArray = np.zeros(len(paramsComb))

    # Iterate over combinations of hyperparams
    for j,params in enumerate(paramsComb):
    
        lambda_reg,gamma = params
        
        # store the squared error for each fold of the cross-validation
        sqErrArray = np.zeros(x.shape[0])
            
        # Do LOO cross-validation
        for i in range(x.shape[0]):
            
            # split into train/test sets
            xTrain,xTest = np.delete(x,i)[:,None],x[i][:,None]
            yTrain,yTest = np.delete(y,i)[:,None],y[i][:,None]
                
            # get a predictor function for the fit model
            f = getPredictor(xTrain,yTrain,lambda x,y : RBF(x,y,gamma),lambda_reg)
            
            sqErrArray[i] = (f(xTest)-yTest)**2
         
        # compute the score of the hyperparams as the mean of squared errors from LOO CV
        scoreArray[j] = np.mean(sqErrArray)
        # print('Done with param combination {}'.format(j))
        
    # Select the best hyperparams
    bestInd = np.argmin(scoreArray)
    lambda_reg,gamma = paramsComb[bestInd]
        
    # get a predictor function for the fit model
    fStar = getPredictor(x,y,lambda x,y : RBF(x,y,gamma),lambda_reg)
    
    return fStar,lambda_reg,gamma

def fitPredictorPoly(x,y,k):
    
    # Choose param space for lambda_reg, kernel hyper param
    lambda_regList = [10**-z for z in range(0,10)]
    dList = list(range(1,21))
    
    paramsComb = list(itertools.product(lambda_regList,dList))
    scoreArray = np.zeros(len(paramsComb))
    fList = []
    # Iterate over combinations of hyperparams
    for j,params in enumerate(paramsComb):
    
        lambda_reg,d = params
        
        # store the squared error for each fold of the cross-validation
        sqErrArray = np.zeros(x.shape[0])
            
        # Do LOO cross-validation
        for i in range(x.shape[0]):
            
            # split into train/test sets
            xTrain,xTest = np.delete(x,i)[:,None],x[i][:,None]
            yTrain,yTest = np.delete(y,i)[:,None],y[i][:,None]
            
            # get a predictor function for the fit model
            f = getPredictor(xTrain,yTrain,lambda x,y : Poly(x,y,d),lambda_reg)
            fList.append(f)
            
            sqErrArray[i] = (f(xTest)-yTest)**2
         
        # compute the score of the hyperparams as the mean of squared errors from LOO CV
        scoreArray[j] = np.mean(sqErrArray)
        # print('Done with param combination {}'.format(j))
    
    # Select the best hyperparams
    bestInd = np.argmin(scoreArray)
    lambda_reg,d = paramsComb[bestInd]
        
    # get a predictor function for the fit model
    fStar = getPredictor(xTrain,yTrain,lambda x,y : Poly(x,y,d),lambda_reg)
    
    return fStar,lambda_reg,d

def plot(fHat,x,y,percL=None,percH=None):
    xGrid = np.linspace(0,1,500)[:,None]
    yTrue = f(xGrid)
    
    fig,ax = plt.subplots(1)
    ax.scatter(x,y,label='Data')
    ax.plot(xGrid,yTrue,label='f')
    ax.plot(xGrid,fHat(xGrid),label='fHat')
    
    if percL is not None and percH is not None:
        # ax.plot(xGrid,percL,label='.05 quantile curve',color='red',ls='--')
        # ax.plot(xGrid,percH,label='.95 quantile curve',color='red',ls='--')
        ax.fill_between(xGrid.flatten(),percL,percH,label='95% confidence region for fHat',color='green',alpha=.3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    ax.set_ylim((-5,10))
    ax.legend()

def runKernelRegression(n):
    
    # Fit params for RBF kernel
    x,yTrue,y = generateData(n=30)
    f_rbf,lambda_reg_rbf,gamma = fitPredictorRBF(x, y, RBF)
    
    print('RBF kernel:')
    print('lambda: {}'.format(lambda_reg_rbf)) #1e-06
    print('gamma: {}'.format(gamma)) #25.108302726288194
    
    # Fit params for polynomial kernel
    f_poly,lambda_reg_poly,d = fitPredictorPoly(x, y, Poly)
    
    print('Polynomial kernel:')
    print('lambda: {}'.format(lambda_reg_poly)) #1e-9
    print('d: {}'.format(d)) #17
    
    plot(f_rbf,x,y)
    plot(f_poly,x,y)
    
    # part c
    n = 30
    B = 300
    gridRes = 500
    
    xGrid = np.linspace(0,1,gridRes)[:,None]
    rbfEst = np.zeros((B,gridRes))
    polyEst = np.zeros((B,gridRes))
    
    bootstrapInds = np.random.choice(range(n),size=(B,30))
    x_b,y_b = x[bootstrapInds],y[bootstrapInds]
    
    for b in range(B):
        
        f_b_rbf = getPredictor(x_b[b], y_b[b], lambda x,y : RBF(x,y,gamma), lambda_reg_rbf)
        f_b_poly = getPredictor(x_b[b], y_b[b], lambda x,y : Poly(x,y,d), lambda_reg_poly)
        
        rbfEst[b,:] = f_b_rbf(xGrid).squeeze()
        polyEst[b,:] = f_b_poly(xGrid).squeeze()
    
    low,high = .05,.95
    
    rbfQuantileL = np.quantile(rbfEst,low,axis=0,interpolation='lower')
    rbfQuantileH = np.quantile(rbfEst,high,axis=0,interpolation='lower')
    polyQuantileL = np.quantile(polyEst,low,axis=0,interpolation='lower')
    polyQuantileH = np.quantile(polyEst,high,axis=0,interpolation='lower')
    
    plot(f_rbf, x, y, rbfQuantileL, rbfQuantileH)
    plot(f_poly, x, y, polyQuantileL, polyQuantileH)