#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:18:45 2020

@author: nick
"""

import csv
import numpy as np
import torch as t
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# code given in the homework spec for loading data
data = []

with open('data/u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
        
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])


data = np.array(data)

num_observations = len(data) # num_observations = 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

# end code from homework spec

#train,test = t.tensor(train),t.tensor(test)
# colNames = ['user','item','rating']
# trainDF,testDF = pd.DataFrame(train,columns=colNames),pd.DataFrame(test,columns=colNames)

def error(Rhat,ref):
    
    if type(Rhat) is np.ndarray:
        Rhat = t.tensor(Rhat)
    
    MSE = t.mean((Rhat[ref[:,1],ref[:,0]] - ref[:,2])**2)
    return MSE
        
def getMaskedTrainMatrix():
    # create a mask for user,item combinations we have data for
    mask = np.ones((num_users,num_items))
    mask[train[:,0],train[:,1]] = 0
    
    # create a data matrix indexed by (user,item)
    Xtrain = np.zeros((num_users,num_items))
    Xtrain[train[:,0],train[:,1]] = train[:,2]
    
    # mask missing values
    Xtrain = np.ma.masked_array(Xtrain,mask)
    
    return Xtrain

# part a
def a():
    
    Xtrain = getMaskedTrainMatrix()
    
    # compute mean for each movie based on the data we have
    mu = np.mean(Xtrain,axis=0)
    
    # fill missing values (movies for which we have no ratings) with average score for all movies
    mu = mu.filled(Xtrain.mean()) 
    
    # the rank one estiamtor is just a tiling of mu (each column is mu)
    Rhat = np.repeat(mu[:,None],repeats=num_users,axis=-1)
    
    # get MSE of the estimator
    MSE = error(Rhat,test)
    print('Mean squared error of the estimator from part a.: {}'.format(MSE))
    
    return Rhat,MSE

Rhat,MSE = a()

# part b
def b():
    
    Xtrain = getMaskedTrainMatrix()
    
    # fill masked data matrix with 0 for unobserved values
    Xtrain = Xtrain.filled(0).T
    
    RhatList = []
    trainErrList = []
    testErrList = []
    dVals = (1,2,5,10,20,50)
    for d in dVals:
        
        # compute top d singular vectors/values
        U,S,VT = svds(Xtrain,d)
    
        # rank-d approximation is given by USVT
        Rhat = U @ np.diag(S) @ VT
        RhatList.append(Rhat)
    
        # compute the train and test error of the estimator
        trainErrList.append(error(Rhat,train))
        testErrList.append(error(Rhat,test))
        
    # plot train,test errors as a function of d
    
    # Note: these results don't seem quite right. Why does test error start increasing for large d?
    # Also, why is the MSE so much higher than part a? Seems counterintuitive.
    fig,ax = plt.subplots(1)
    ax.plot(dVals,trainErrList,label='Train Error')
    ax.plot(dVals,testErrList,label='Test Error')
    ax.legend()
    ax.set_title('SVD Losses')
    
    return

b()

def lossFn(U,V,lambda_reg,ref=train):
    """
    Compute the loss of the vectors U in R^d representing users and the vector V in R^d representing items.

    Parameters
    ----------
    U : torch tensor
        Rows are the d-dimensional representations of users
    VT : torch tensor
        Rows are the d-dimensional representations of items
    lambda_reg : float
        Regularization constant between 0 and 1

    Returns
    -------
    loss : torch tensor
        Loss of the learned representations

    """
    if type(U)==np.ndarray:
        U = t.tensor(U)
        V = t.tensor(V)
    
    Rhat = U @ V.T
    
    loss = error(Rhat,ref) + lambda_reg * (t.sum(t.norm(U,dim=1)**2) + t.sum(t.norm(V,dim=1)**2))
    
    return loss

def alternatingMin(d,sigma,lambda_reg):
    
    print('d={}'.format(d))
    
    U = sigma*np.random.rand(num_items,d)
    V = sigma*np.random.rand(num_users,d)

    # iteratively minimize loss until loss converges
    _loss = 999999
    loss = _loss
    delta = 999999
    while np.abs(delta) > 1e-3 * loss:
        
        # define some helper functions
        get_j = lambda j : train[:,0]==j
        get_i = lambda i : train[:,1]==i
        
        A_array = np.zeros((num_items,d,d))
        B_array = np.zeros((num_items,d))
        
        RV = train[:,2,None]*V[train[:,0]]
        
        for i in range(num_items):
            
            idx = get_i(i)
            
            # compute sum of outer product of rows of V plus diagonal regularization matrix
            A = np.sum(V[train[idx,0],:,None]*V[train[idx,0],None],axis=0) + lambda_reg*np.eye(d)
            B = np.sum(RV[idx],axis=0)
            
            A_array[i] = A
            B_array[i] = B.T
            
        # solve for U which minimizes loss for fixed V
        U = np.linalg.solve(A_array,B_array)
            
        A_array = np.zeros((num_users,d,d))
        B_array = np.zeros((num_users,d))
        
        RU = train[:,2,None]*U[train[:,1]]
        
        for j in range(num_users):
            
            idx = get_j(j)
            
            # compute sum of outer product of rows of U plus diagonal regularization matrix
            A = np.sum(U[train[idx,1],:,None]*U[train[idx,1],None],axis=0) + lambda_reg*np.eye(d)
            B = np.sum(RU[idx],axis=0)
            
            A_array[j] = A
            B_array[j] = B.T
            
        # solve for V which minimizes loss for fixed U
        V = np.linalg.solve(A_array,B_array)
    
        # compute the loss
        loss = lossFn(U,V,lambda_reg)
        
        # compute change in loss
        delta = loss - _loss
        _loss = loss
        print('Loss: {}'.format(loss))
        
    # compute MSE
    Rhat = U @ V.T
    trainMSE,testMSE = error(Rhat,train),error(Rhat,test)

    return U,V,trainMSE,testMSE

# part c
def c(sigma,lambda_reg):
    
    UList = []
    VList = []
    trainErrList = []
    testErrList = []
    dVals = (1,2,5,10,20,50)
    
    # R = getMaskedTrainMatrix()
    for d in dVals:
    
        U,V,trainMSE,testMSE = alternatingMin(d, sigma, lambda_reg)
            
        UList.append(U)
        VList.append(V)
        trainErrList.append(trainMSE)
        testErrList.append(testMSE)
        
    fig,ax = plt.subplots(1)
    ax.plot(dVals,trainErrList,label='Train Error')
    ax.plot(dVals,testErrList,label='Test Error')
    ax.set_xlabel('d')
    ax.set_ylabel('Error')
    ax.legend()
    ax.set_title('Iterative Minimization Losses')
        
    return UList,VList,trainErrList,testErrList

# UList,VList,trainErrList,testErrList = c(sigma=5,lambda_reg=.4) 

# part d

# split dataset to get a validation set
_train = t.tensor(train)
trainInd = int(.8*_train.shape[0])
_train,_validation = _train[:trainInd],_train[trainInd:]
_test = t.tensor(test)

def sgdMin(d, sigma, lambda_reg, batchSize, lr, beta, validation=False):
    
    extraData = _validation if validation else _test
    
    # randomly initialize U,V
    U = t.tensor(sigma*np.random.rand(num_items,d))
    V = t.tensor(sigma*np.random.rand(num_users,d))
    
    # need grad for optimization
    U.requires_grad = True
    U.retain_grad()
    V.requires_grad = True
    V.retain_grad()
    
    counter = 0
    _loss = t.tensor(999999)
    loss = _loss
    delta = t.tensor(999)
    i = 0
    deltaList = []
    minLoss = 999999
    while t.abs(t.tensor(delta)) > 1e-5 * loss:
    
        # check if we need to decay learn rate
        if counter > _train.size(0):
            lr = beta * lr
            counter = 0
    
        # randomly select batch of training data to compute error
        perm = t.randperm(_train.size(0))
        idx = perm[:batchSize]
        batch = _train[idx]
        
        # compute gradient
        loss = lossFn(U,V,lambda_reg,ref=batch)
        loss.backward()

        if loss < minLoss:
            minLoss = loss
        
        # take a step
        with t.no_grad():
            U += - lr * U.grad
            V +=  - lr * V.grad

        # zero gradients
        U.grad.zero_()
        V.grad.zero_()
        
        delta = loss - _loss
        deltaList.append(t.abs(delta))
            
        #print('loss: {}'.format(loss))
        _loss  = loss
        
        counter += batchSize
        
        i += 1
    
    # compute MSE
    Rhat = U @ V.T
    trainMSE,extraMSE = error(Rhat,_train),error(Rhat,extraData)
    
    return U,V,trainMSE,extraMSE

def d():
    
    nPoints = 100
    sigmaSpace = 3*t.rand(size=(nPoints,))
    lambdaSpace = t.rand(size=(nPoints,))
    # batchPoss = t.tensor([64,128,256,512,1024,2048])
    # batchInd = t.randint(0,4,size=(nPoints,))
    # batchSpace = batchPoss[batchInd]
    batchSpace = t.randint(64,5096,size=(nPoints,))
    lrSpace = t.randint(1,5,size=(nPoints,))
    lrSpace = 10** -lrSpace.float()
    betaSpace = .25*t.rand(size=(nPoints,)) + .75
    paramSpaces = [list(space) for space in (sigmaSpace,lambdaSpace,batchSpace,lrSpace,betaSpace)]
    paramCombList = list(zip(*paramSpaces))
    dVals = (1,2,5,10,20,50)
    
    bestParamsList,bestTrainErrList,bestTestErrList = [],[],[]
    for dVal in dVals:
        
        UList_d,VList_d,trainErrList_d,valErrList_d = [],[],[],[]
        
        for idx,params in enumerate(paramCombList):
            
            U,V,trainErr,valErr = sgdMin(dVal,*params,validation=True)
        
            print('Done with param combination {} for d={}! Error is {}'.format(idx,dVal,valErr))    
            
            if not (t.isnan(valErr) or t.isnan(trainErr)):
                trainErrList_d.append(trainErr.detach())
                valErrList_d.append(valErr.detach())
                UList_d.append(U)
                VList_d.append(V)
                
        print('Done with d={}!'.format(dVal))
            
        # plot the test errors
        fig,ax = plt.subplots(1)
        ax.violinplot(t.tensor(valErrList_d)[:,None].T)
        ax.set_ylabel('Validation Error')
        ax.set_title('Distribution of validation errors for random hyperparameter search\n d={}'.format(dVal))
        
        bestParamsInd = t.argmin(t.tensor(valErrList_d))
        bestParams = paramCombList[bestParamsInd]
        bestParamsList.append(bestParams)
        bestTrainErrList.append(trainErrList_d[bestParamsInd])
        
        # compute test error for the best parameters
        U,V = UList_d[bestParamsInd],VList_d[bestParamsInd]
        bestTestErrList.append(error(U @ V.T,_test))
        
    fig,ax = plt.subplots(1)
    ax.plot(dVals,bestTrainErrList,label='Train Error')
    ax.plot(dVals,bestTestErrList,label='Test Error')
    ax.set_xlabel('d')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.set_title('SGD Minimization Losses')
    
    for dVal,params in zip(dVals,bestParamsList):
        print('best params for d={}:'.format(dVal))
        print(params)
    
d()