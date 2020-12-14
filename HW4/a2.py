#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:57:51 2020

@author: nick
"""
from mnist import MNIST
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt

mndata = MNIST('./data/')
X_train, y_train = map(np.array, mndata.load_training())
X_test, y_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

# Conver to tensors
X_train,y_train = torch.tensor(X_train).float(),torch.tensor(y_train).long()
X_test,y_test = torch.tensor(X_test).float(),torch.tensor(y_test).long()

digits = X_train[[1,3,5,7,9,11,13,15,17,22]]

class AutoEncoder(torch.nn.Sequential):
    
    def __init__(self,layers,linear=True):
        
        # for each layer specified, reverse the dimensions and add a new layer
        # for the decoder
        _layers = copy.copy(layers)
        for layer in _layers[::-1]:
            if not (type(layer) is torch.nn.ReLU):
                newLayer = type(layer)(layer.out_features,layer.in_features)
                layers.append(newLayer)
                if not linear:
                    layers.append(torch.nn.ReLU())
        
        # construct the neural net
        super(AutoEncoder,self).__init__(*layers)
        
    def train(self,X,nEpochs=250,lr=.05,stochastic=False,batchSize=5000):
        
        opt = torch.optim.Adam(params=self.parameters(),lr=lr)
        lossfn = torch.nn.MSELoss()
        
        for epoch in range(nEpochs):
            
            opt.zero_grad()
            
            if stochastic:
                batchInd = torch.randperm(X.shape[0])[:batchSize]
                _X = X[batchInd]
            else:
                _X = X
                
            output = self.forward(_X)
            
            loss = lossfn(_X,output)
            loss.backward()
            print('Loss at epoch {}: {}'.format(epoch,loss))            
            
            opt.step()
            
        return loss.detach()

def a():
    
    # create and train the neural networks
    nnList = []
    finalLossList = []
    reconstructedDigitsList = []
    for h in (32,64,128):
        nn = AutoEncoder([torch.nn.Linear(X_train.shape[-1],h),])
        finalLoss = nn.train(X_train, nEpochs=500, lr=.005, stochastic=True)
        
        nnList.append(nn)
        finalLossList.append(finalLoss)
        print('Final loss for h={}: {}'.format(h,finalLoss)) # .0175, 0.0096, 0.0049
        
        # run each digit through the neural network
        reconstrDigits = nn(digits).detach()
        reconstructedDigitsList.append(reconstrDigits)
    
        # show the digits
        fig,axes = plt.subplots(2,5)
        fig.suptitle('MNIST Reconstructions for h={}'.format(h))
        _axes = []
        for item in axes:
            _axes += list(item)
        for i,ax in enumerate(_axes):
            ax.imshow(reconstrDigits[i].reshape((28,28)),cmap='Greys')
            ax.set_title('Digit: {}'.format(i))
            
    return nnList,finalLossList,reconstructedDigitsList
      
def b():
    
    # create and train the neural networks
    nnList = []
    finalLossList = []
    reconstructedDigitsList = []
    for h in (32,64,128):
        nn = AutoEncoder([torch.nn.Linear(X_train.shape[-1],h),
                          torch.nn.ReLU()],linear=False)
        finalLoss = nn.train(X_train, nEpochs=1000, lr=.001, stochastic=True, batchSize=10000)
        
        nnList.append(nn)
        finalLossList.append(finalLoss)
        print('Final loss for h={}: {}'.format(h,finalLoss)) # 0.0246, 0.0092, 0.0047
        
        # run each digit through the neural network
        reconstrDigits = nn(digits).detach()
        reconstructedDigitsList.append(reconstrDigits)
    
        # show the digits
        fig,axes = plt.subplots(2,5)
        fig.suptitle('MNIST Reconstructions for h={}'.format(h))
        _axes = []
        for item in axes:
            _axes += list(item)
        for i,ax in enumerate(_axes):
            ax.imshow(reconstrDigits[i].reshape((28,28)),cmap='Greys')
            ax.set_title('Digit: {}'.format(i))
            
    return nnList,finalLossList,reconstructedDigitsList

def c(nnA,nnB):
    
    lossFn = torch.nn.MSELoss()
    errorA = lossFn(nnA(X_test),X_test)
    errorB = lossFn(nnB(X_test),X_test)
    
    return errorA,errorB

torch.manual_seed(42)
resultsA = a()
resultsB = b()
mseA,mseB = c(resultsA[0][-1],resultsB[0][-1]) # .0048, .0046