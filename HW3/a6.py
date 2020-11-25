#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:17:59 2020

@author: nick
"""

import torch
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST('./data/')
X_train, y_train = map(np.array, mndata.load_training())
X_test, y_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

# y_train = np.array([y_train==i for i in range(0,10)]).astype(int).T
# y_test = np.array([y_test==i for i in range(0,10)]).astype(int).T

# Conver to tensors
X_train,y_train = torch.tensor(X_train).float(),torch.tensor(y_train).long()
X_test,y_test = torch.tensor(X_test).float(),torch.tensor(y_test).long()

# Create a wide, shallow NN and train on MNIST data
def a():
    
    d = X_train.shape[1] # 784
    h = 64
    k = 10 #y_train.shape[1] # 10
    
    alpha = 1/np.sqrt(d)
    W0 = torch.FloatTensor(h, d).uniform_(-alpha,alpha)
    b0 = torch.FloatTensor(h, 1).uniform_(-alpha,alpha)
    
    W1 = torch.FloatTensor(k, h).uniform_(-alpha,alpha)
    b1 = torch.FloatTensor(k, 1).uniform_(-alpha,alpha)
    
    params = [W0,b0,W1,b1]
    for param in params:
        param.requires_grad=True
        
    # Create a function which pushes x through the NN
    def F(x,W0,b0,W1,b1):
        
        layer1 = torch.nn.functional.relu(W0.matmul(x.T)+b0)
        output = W1.matmul(layer1)+b1
        return output

    # Train the NN using ADAM optimizer
    lossList = []
    eta = .001
    batchSize = 10000
    opt = torch.optim.Adam([W0,b0,W1,b1], lr=eta)
    predAccuracy = 0
    epoch = 0
    while predAccuracy<.99:
        
        # zero gradient before next iteration
        opt.zero_grad()
        
        print('Starting epoch {}!'.format(epoch))
        
        # Select a batch of training data
        batchIndices = torch.randint(0, X_train.size()[0], size=(batchSize,))
        X_batch,y_batch = X_train[batchIndices],y_train[batchIndices]
        
        yHat_batch = F(X_batch,W0,b0,W1,b1)
        
        loss = torch.nn.functional.cross_entropy(yHat_batch.T, y_batch)
        lossList.append(loss)
        
        loss.backward()
        
        # take a step with ADAM optimizer
        opt.step()
        
        # compute prediction accuracy
        yHat = F(X_train,W0,b0,W1,b1)
        predLabels = torch.argmax(yHat,dim=0)
        predAccuracy = torch.sum(predLabels==y_train)/float(y_train.size()[0])
        print('Prediction accuracy: {}%'.format(predAccuracy*100))
        
        epoch += 1
        
    return [W0,b0,W1,b1],lossList,F
        
# Create a narrow, deep NN and train on MNIST data
def b():
    
    d = X_train.shape[1] # 784
    h = 32
    k = 10
    
    alpha = 1/np.sqrt(X_train.shape[1])
    W0 = torch.FloatTensor(h, d).uniform_(-alpha,alpha)
    b0 = torch.FloatTensor(h, 1).uniform_(-alpha,alpha)
    
    W1 = torch.FloatTensor(h, h).uniform_(-alpha,alpha)
    b1 = torch.FloatTensor(h, 1).uniform_(-alpha,alpha)
    
    W2 = torch.FloatTensor(k, h).uniform_(-alpha,alpha)
    b2 = torch.FloatTensor(k, 1).uniform_(-alpha,alpha)
    
    params = [W0,b0,W1,b1,W2,b2]
    for param in params:
        param.requires_grad=True
    
    # Create a function which pushes x through the NN
    def F(x,W0,b0,W1,b1,W2,b2):
        
        layer1 = torch.nn.functional.relu(W0.matmul(x.T)+b0)
        layer2 = torch.nn.functional.relu(W1.matmul(layer1)+b1)
        output = W2.matmul(layer2)+b2
        return output

     # Train the NN using ADAM optimizer
    lossList = []
    eta = .001
    batchSize = 10000
    opt = torch.optim.Adam([W0,b0,W1,b1,W2,b2], lr=eta)
    predAccuracy = 0
    epoch = 0
    while predAccuracy<.99:
        
        # zero gradient before next iteration
        opt.zero_grad()
        
        print('Starting epoch {}!'.format(epoch))
        
        # Select a batch of training data
        batchIndices = torch.randint(0, X_train.size()[0], size=(batchSize,))
        X_batch,y_batch = X_train[batchIndices],y_train[batchIndices]
        
        yHat_batch = F(X_batch,W0,b0,W1,b1,W2,b2)
        
        loss = torch.nn.functional.cross_entropy(yHat_batch.T, y_batch)
        lossList.append(loss)
        
        loss.backward()
        
        # take a step with ADAM optimizer
        opt.step()
        
        # compute prediction accuracy
        yHat = F(X_train,W0,b0,W1,b1,W2,b2)
        predLabels = torch.argmax(yHat,dim=0)
        predAccuracy = torch.sum(predLabels==y_train)/float(y_train.size()[0])
        print('Prediction accuracy: {}%'.format(predAccuracy*100))
        
        epoch += 1
        
    return [W0,b0,W1,b1,W2,b2],lossList,F

def runPartA():

    # training this took 1634 epochs    
    # model has 784*64 + 64 + 10*64 + 10 = 50890 parameters

    # Train the wide, shallow NN
    weights,losses,F = a()
    
    # Compute test accuracy and loss
    y_pred = F(X_test,*weights)
    testLoss = torch.nn.functional.cross_entropy(y_pred.T, y_test) #0.09483137726783752
    testLabels = torch.argmax(y_pred,dim=0) 
    testAccuracy = torch.sum(testLabels==y_test)/float(y_test.size()[0]) #0.9717000126838684

    print('Test loss: {}'.format(testLoss))
    print('Test accuracy: {}'.format(testAccuracy))
    
    fig,ax = plt.subplots(1)
    ax.plot(range(len(losses)),losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')

def runPartB():

    # training this took 2282 epochs
    # model has 784*32 + 32 + 32*32 + 32 + 32*10 + 10 = 26506 parameters
    
    # Train the wide, shallow NN
    weights,losses,F = b()
    
    # Compute test accuracy and loss
    y_pred = F(X_test,*weights)
    testLoss = torch.nn.functional.cross_entropy(y_pred.T, y_test) #0.12587575614452362
    testLabels = torch.argmax(y_pred,dim=0) 
    testAccuracy = torch.sum(testLabels==y_test)/float(y_test.size()[0]) #0.9668999910354614

    print('Test loss: {}'.format(testLoss))
    print('Test accuracy: {}'.format(testAccuracy))
    
    fig,ax = plt.subplots(1)
    ax.plot(range(len(losses)),losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')


