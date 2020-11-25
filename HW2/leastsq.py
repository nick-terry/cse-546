#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:59:19 2020

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt

def generateData(n,d):
    
    iVect = np.array(range(1,n+1)).reshape((n,1))
    iVect = np.sqrt(iVect%d + 1)

    X = np.zeros(shape=(n,d))
    for i in range(1,n+1):
        x = np.zeros(d)
        x[i%d] = 1
        X[i-1,:] = iVect[i-1]*x
    
    # We assume here that beta^* is 0
    Y = np.random.normal(0,1,size=(n,1))
    
    return X,Y

np.random.seed(42)
n = 20000
d = 10000
X,Y = generateData(n, d)

# Compute (X^T@X)^-1 analytically
XTXI = np.zeros(shape=(d,d))
for j in range(1,d+1):
    XTXI[j-1,j-1] = 1/(np.floor((n-j+1)/d)*(j%d+1))
    
# Compute betaHat
betaHat = XTXI@X.T@Y

# Compute confidence intervals
delta = .95
hw = np.sqrt(2*np.diag(XTXI)*np.log(2/delta))

# Determine how many points are outside confidence interval
countOutside = np.sum(np.abs(betaHat)>hw.reshape(hw.size,1))
print('{} entries of betaHat outside of the confidence interval'.format(countOutside))

fig,ax = plt.subplots(1)
ax.scatter(range(1,d+1),betaHat)
ax.plot(range(1,d+1),hw,color='orange',label='upper {}% confidence bound'.format(delta*100))
ax.plot(range(1,d+1),-hw,color='orange',label='lower {}% confidence bound'.format(delta*100))
ax.set_xlabel('j')
ax.set_ylabel('betaHat_j')
ax.legend()

