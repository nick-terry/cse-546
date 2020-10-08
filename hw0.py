#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 08:54:58 2020

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt

#A.11
A = np.array([[0,2,4],
              [2,4,2],
              [3,3,1]])
b = np.array([-4,-2,-2]).reshape((3,1))
c = np.array([1,1,1]).reshape((3,1))

#a)
print('a)')
Ainv = np.linalg.inv(A)
print(Ainv)
print('\n')
#b)
print('b)')
print(Ainv @ b)
print('\n')
print(A @ c)
print('\n')

#A.12
np.random.seed(42)

#a)
#From A.6, compute the number of samples needed.
n = 200**2
Z = np.random.randn(n)
fig,ax = plt.subplots(1)
ax.step(sorted(Z),np.arange(1,n+1)/float(n),label='Gaussian')

#Compute Fhat_n
x = np.linspace(-3,3,num=1000)
f = lambda x : np.mean(Z<=x)
Fhat_n = np.array([f(_x) for _x in x])
ax.plot(x,Fhat_n,label='Fhat_n')

#b)
for k in [1,8,64,512]:
    Yk = np.sum(np.sign(np.random.randn(n, k))*np.sqrt(1./k), axis=1)
    ax.step(sorted(Yk),np.arange(1,n+1)/float(n),label=str(k))

ax.set_xlim(-3,3)
ax.set_xlabel('Observations')
ax.set_ylabel('Probability')
ax.legend()