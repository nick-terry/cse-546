#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:43:40 2020

@author: nick
"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST('./data/')
X_train, y_train = map(np.array, mndata.load_training())
X_test, y_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

# compute mean of training set
mu = np.mean(X_train,axis=0)

# compute sample covar matrix
diff = X_train - mu
sigma = diff.T @ diff / X_train.shape[0]

Lambda, V = np.linalg.eig(sigma)

# print some eigenvalues
for i in (1,2,10,30,50):
    print('lambda {}: {}'.format(i,Lambda[i-1]))
    
print('Sum of eigenvalues: {}'.format(np.sum(Lambda)))

# part c
def changeBasis(X,basis):
    
    # project X onto basis w/ inner products
    proj = X @ basis
    
    # compute reconstruction as linear comb of eigenvectors
    reconstr = proj @ basis.T
    
    # compute reconstruction error
    MSE = np.mean(np.sum((reconstr - X)**2,axis=1),axis=0)
    
    return reconstr,MSE

eigValsList = []
trainErrList = []
testErrList = []

for k in range(1,101):
    
    eigValsList.append(1-np.sum(Lambda[:k])/np.sum(Lambda))
    trainProj,trainMSE = changeBasis(X_train, V[:,:k])
    testProj,testMSE = changeBasis(X_test, V[:,:k])
    
    trainErrList.append(trainMSE)
    testErrList.append(testMSE)
    
fig,ax = plt.subplots(1)
ax.plot(range(1,101),trainErrList,label='Training Set Reconstruction Error')
ax.plot(range(1,101),testErrList,label='Test Set Reconstruction Error')
ax.set_xlabel('k')
ax.set_ylabel('Reconstruction MSE')
ax.legend()

fig,ax = plt.subplots(1)
ax.plot(range(1,101),eigValsList)
ax.set_xlabel('k')
ax.set_ylabel('Sum of first k eigenvalues')

'''
lambda 1: (5.116787728342082+0j)
lambda 2: (3.741328478864825+0j)
lambda 10: (1.242729376417333+0j)
lambda 30: (0.364255720278891+0j)
lambda 50: (0.1697084270067276+0j)
Sum of eigenvalues: (52.72503549512702+0j)
'''

# part d
fig,axes = plt.subplots(2,5)
axes = np.array(axes).flatten()
    
for i,ax in enumerate(axes):
    ax.imshow(np.real(V[:,i]).reshape(28,28),cmap='Greys')
    
# part e
imageInds = (5,13,15) # indices for images of the digits 2,6,7 resp.

for imageInd in imageInds:
    image = X_train[None,imageInd]
    
    reconstrList = [image.reshape(28,28),]
    for k in (5,15,40,100):
        reconstr,MSE = changeBasis(image, np.real(V[:,:k]))
        reconstrList.append(reconstr.reshape(28,28))
        
    fig,axes = plt.subplots(1,5)
    axes = np.array(axes).flatten()
    for i,ax in enumerate(axes):
        ax.imshow(reconstrList[i],cmap='Greys')