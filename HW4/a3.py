#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:14:59 2020

@author: nick
"""

import torch
import torchvision
import os

T = torchvision.transforms.Compose([torchvision.transforms.Pad((112,)*4),
                                    torchvision.transforms.ToTensor()])
dataPath = os.path.join(os.getcwd(),'data')
dataset = torchvision.datasets.CIFAR10(dataPath,transform=T)
train,validation = torch.utils.data.random_split(dataset,
                                                 [40000,10000],
                                                 torch.Generator().manual_seed(42))
test = torchvision.datasets.CIFAR10(dataPath,
                                          transform=T,
                                          train=False)

trainL = torch.utils.data.DataLoader(train,batch_size=2000)
validL = torch.utils.data.DataLoader(validation,batch_size=10000)
testL = torch.utils.data.DataLoader(test,batch_size=10000)

def a():
    
    # load AlexNet
    model = torchvision.models.alexnet(pretrained=True)
    # freeze params
    for param in model.parameters():
        param.requires_grad = False
    # replace last layer with Linear(4096,10)
    model.classifier[-1] = torch.nn.Linear(4096,10)
    
    # train the new output layer
    nEpochs = 20
    lossFn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params=model.parameters(),lr=.05)
    
    trainLosses = torch.zeros((nEpochs,))
    validationLosses = torch.zeros((nEpochs,))
    validationAcc = torch.zeros((nEpochs,))
    
    for epoch in range(nEpochs):
        
        opt.zero_grad()
        
        tLosses = []
        for features,targets in trainL:
            
            features,targets = features,targets
            
            loss = lossFn(model(features),targets)
            loss.backward()
            tLosses.append(loss.detach())
        
        tloss = torch.mean(torch.tensor(tLosses))
        print('Loss at epoch {}: {}'.format(epoch,tloss))        
        
        for features,targets in validL:
            
            features,targets = features,targets
            
            # get validation loss
            vout = model(features)
            predLabels = torch.argmax(vout,dim=1)
            
            vloss = lossFn(vout,targets)
            vacc = torch.sum(targets==predLabels)/float(targets.shape[0])*100
            
        trainLosses[epoch] = tloss.detach()
        validationLosses[epoch] = vloss.detach()
        validationAcc[epoch] = vacc
        
        opt.step()
        
    return model,trainLosses,validationLosses,validationAcc

results = a()