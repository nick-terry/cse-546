#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:09:41 2020

@author: nick
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# B.1 
# d.
n = 256
sigmaSq = 1.0
f = lambda x : 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

x = np.array(range(1,n+1))/n
eta = np.random.normal(0,sigmaSq,size=256)
y = f(x) + eta

fig,ax = plt.subplots(1)
ax.plot(x,y,label='f')

mTup = (1,2,4,8,16,32)
empErrArr = np.zeros(shape=(len(mTup),))
avgSqBiasArr = np.zeros_like(empErrArr)
avgVarArr = np.zeros_like(empErrArr)
avgErrArr = np.zeros_like(empErrArr)

## Something doesnt work here
for k,m in enumerate(mTup):
    
    c = np.zeros(shape=(int(n/m),1))
    for j in range(1,int(n/m)+1):
        
        c[j-1] = np.sum(y[(j-1)*m:j*m])/m

    jArr = np.array(range(1,int(n/m)+1))
    lb = (jArr-1)*m/n
    ub = jArr*m/n
    fhat = np.zeros_like(x)
    for i in range(n):
        fhat[i] = np.dot((lb<x[i]) * (x[i]<=ub),c)

    empErrArr[k] = np.mean((f(x)-fhat)**2)
    
    bias = np.zeros(shape=(int(n/m),1))
    for j in range(1,int(n/m)+1):
        fbar = np.mean(f(x)[(j-1)*m:j*m])
        bias[j-1] = np.sum((fbar-f(x[(j-1)*m:j*m+1]))**2)
    
    avgSqBiasArr[k] = np.sum(bias)/n
    avgVarArr[k] = sigmaSq/m
    avgErrArr[k] = avgVarArr[k] + avgSqBiasArr[k]
      
    ax.plot(x,fhat,label='m={}'.format(m))
    
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

fig2,ax2 = plt.subplots(1)
ax2.plot(mTup,empErrArr,label='Average Empirical Error')
ax2.plot(mTup,avgSqBiasArr,label='Average Bias-Squared')
ax2.plot(mTup,avgVarArr,label='Average Variance')
ax2.plot(mTup,avgErrArr,label='Average Error')
ax2.legend()
ax2.set_xlabel('m')
ax2.set_ylabel('Error/Bias/Variance')