#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:02:50 2020

@author: nick
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

mndata = MNIST('./data/')
X_train, y_train = map(np.array, mndata.load_training())
X_test, y_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

y_train = np.array([y_train==i for i in range(0,10)]).astype(int).T
y_test = np.array([y_test==i for i in range(0,10)]).astype(int).T

# Conver to tensors
X_train,y_train = torch.tensor(X_train).float(),torch.tensor(y_train).long()
X_test,y_test = torch.tensor(X_test).float(),torch.tensor(y_test).long()

np.random.seed(42)

def gradDescent(X_train,y_train,lossFn,step_size,eps=1e-3):
    
    W = torch.zeros(X_train.shape[1], 10, requires_grad=True)
    deltaList = []
    _loss = None
    
    # Create matrix to map one hot encoding to label
    A = torch.tensor(range(10)).reshape(10,1).long()
    
    while len(deltaList) < 10 or torch.mean(torch.tensor(deltaList[-10:])) <= -eps:
        
        # Compute labels with W
        y_hat = torch.matmul(X_train, W)
        
        # compute loss
        loss = lossFn(y_hat, y_train.matmul(A).squeeze())
        if _loss is not None:
            deltaList.append(loss-_loss)
            print(loss-_loss)
        
        # computes derivatives of the loss with respect to W
        loss.backward()
        
        # gradient descent update
        W.data = W.data - step_size * W.grad
        
        # .backward() accumulates gradients into W.grad instead
        # of overwriting, so we need to zero out the weights
        W.grad.zero_()
        
        _loss = loss
        
    return W

def stochasticGradDescent(X_train,y_train,lossFn,batchSize,step_size,eps=.1):
    
    W = torch.zeros(X_train.shape[1], 10, requires_grad=True)
    deltaList = []
    _loss = None
    
    while len(deltaList) < 10 or torch.mean(torch.tensor(deltaList[-10:])) <= -eps:
        
        # Randomly select batchSize observations to compute gradient
        gradInd = np.random.choice(range(X_train.shape[0]),size=batchSize,replace=False)
        
        Xgrad,ygrad = X_train[gradInd,:],y_train[gradInd]
        
        # Compute labels with W
        y_hat = torch.matmul(Xgrad, W)
        
        # compute loss
        loss = lossFn(y_hat, ygrad)
        if _loss is not None:
            deltaList.append(loss-_loss)
            print(loss-_loss)
        
        # computes derivatives of the loss with respect to W
        loss.backward()
        
        # gradient descent update
        W.data = W.data - step_size * W.grad
        
        # .backward() accumulates gradients into W.grad instead
        # of overwriting, so we need to zero out the weights
        W.grad.zero_()
        
        _loss = loss
        
    return W

# Do multinomial logistic regression using the cross-entropy loss function
W1 = gradDescent(X_train,y_train,torch.nn.functional.cross_entropy,step_size=.005)

# Do ridge regression using the J loss function
def J(y_hat,y): 
    
    return .5*torch.sum(torch.norm(y-y_hat,p=2,dim=1)**2)

W2 = stochasticGradDescent(X_train,y_train,J,batchSize=100,step_size=.0005)

# Compute the classifications errors for train/test sets of each model
labels_train = torch.argmax(y_train,dim=1)
labels_test = torch.argmax(y_test,dim=1)

# Multinomial regression
trainPreds1 = torch.argmax(X_train.matmul(W1),dim=1)
classAccTrain1 = torch.sum(labels_train==trainPreds1)/float(labels_train.shape[0])

testPreds1 = torch.argmax(X_test.matmul(W1),dim=1)
classAccTest1 = torch.sum(labels_test==testPreds1)/float(labels_test.shape[0])

print('L(W)')
print('Training data classification accuracy: {}%'.format(torch.round(classAccTrain1*100)))
print('Test data classification accuracy: {}%'.format(torch.round(classAccTest1*100)))

# Ridge regression
trainPreds2 = torch.argmax(X_train.matmul(W2),dim=1)
classAccTrain2 = torch.sum(labels_train==trainPreds2)/float(labels_train.shape[0])

testPreds2 = torch.argmax(X_test.matmul(W2),dim=1)
classAccTest2 = torch.sum(labels_test==testPreds2)/float(labels_test.shape[0])

print('\nJ(W)')
print('Training data classification accuracy: {}%'.format(torch.round(classAccTrain2*100)))
print('Test data classification accuracy: {}%'.format(torch.round(classAccTest2*100)))
