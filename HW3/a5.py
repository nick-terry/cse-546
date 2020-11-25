#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:28:11 2020

@author: nick
"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def loss(X,C,labels):
    return np.mean(np.linalg.norm(X-C[labels],ord=2,axis=1)**2)

def lloyd(X,k):
    
    lossList = []
    
    # initialize the centroids by randomly sampling points from X
    C = np.random.choice(range(X.shape[0]),size=k,replace=False)
    C = X[C]
    
    labels = np.array([np.argmin(np.linalg.norm(X[i]-C,ord=2,axis=1)**2) for i in range(len(X))])
    lossList.append(loss(X,C,labels))
    labels_old = np.zeros_like(labels)
    
    # iterate
    while np.any(labels_old!=labels):
        
        # update centroids
        C = np.array([np.mean(X[labels==i],axis=0) for i in range(k)])
        
        # update labels
        labels_old = labels
        labels = np.array([np.argmin(np.linalg.norm(X[i]-C,ord=2,axis=1)**2) for i in range(len(X))])
        
        lossList.append(loss(X,C,labels))
        
    return C,labels,lossList


mndata = MNIST('./data/')
X_train, y_train = map(np.array, mndata.load_training())
X_test, y_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

y_train = np.array([y_train==i for i in range(0,10)]).astype(int).T
y_test = np.array([y_test==i for i in range(0,10)]).astype(int).T

# run k-means on MNIST for part c
def c():
    C,labels,lossList = lloyd(X_train,k=10)
    
    plt.plot(range(1,len(lossList)),lossList[1:])
    plt.xlabel('Iteration #')
    plt.ylabel('Mean Loss')
    
    fig,axes = plt.subplots(2,5)
    axes = np.array(axes).flatten()
    
    for i,ax in enumerate(axes):
        ax.imshow(C[i].reshape(28,28),cmap='Greys')    
    
# run with varying k for part d
def d():
    
    trainLoss,testLoss = [],[]
    kTup = (2,4,8,16,32,64)
    
    for k in kTup:
        C,labels,lossList = lloyd(X_train,k)
        
        trainLoss.append(lossList[-1])
        
        # get labels for test set
        testLabels = np.array([np.argmin(np.linalg.norm(X_test[i]-C,ord=2,axis=1)**2) for i in range(len(X_test))]) 
        lossRes = loss(X_test,C,testLabels)
        testLoss.append(lossRes)
    
    fig,ax = plt.subplots(1)
    ax.plot(kTup,trainLoss,label='Training Error')
    ax.plot(kTup,testLoss,label='Test Error')
    ax.set_xlabel('k')
    ax.set_ylabel('Error')
    ax.legend()
    
    return ax

d()
