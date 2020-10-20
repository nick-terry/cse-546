#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:17:18 2020

@author: nick
"""
import numpy as np
from mnist import MNIST


mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0
    
def train(X,Y,reg_lambda):
    """
    Train a ridge regression model using closed form solution.

    Parameters
    ----------
    X : n x d numpy array
        Training features
    Y : n x k numpy array
        Training labels
    lambda : float
        Regularization parameter.

    Returns
    -------
    W_hat : d x k numpy array
        The prediction matrix.

    """
    
    #Create one-hot encoding for Y
    _Y = np.array([labels_train==i for i in range(0,10)]).astype(int).T
    
    #Solve the ridge regression using closed form solution
    A = X.T @ X + reg_lambda * np.eye(X.shape[-1])
    B = X.T @ _Y
    W_hat = np.linalg.solve(A,B)
    
    return W_hat

def predict(W,X):
    """
    Create predictions using the ridge regression

    Parameters
    ----------
    W : d x k numpy array
        Prediction matrix.
    X : d x m numpy array
        Inputs for which to predict the labels.

    Returns
    -------
    Y : m x 1 numpy array
        The predicted labels.

    """
    
    result = W.T @ X.T
    
    #Get the index of the largest entry as the class
    Y = np.argmax(result,axis=0)
    
    return Y

if __name__=='__main__':
        
    #Solve ridge regression on MNIST dataset
    W_hat = train(X_train,labels_train,10**-4)
    
    #Predict the labels for each digit
    y_train = predict(W_hat,X_train)
    y_test = predict(W_hat,X_test)
    
    #Compute the training and test error
    training_error = np.mean(y_train!=labels_train)
    test_error = np.mean(y_test!=labels_test)
    
    print('Training Error: {}'.format(training_error))
    print('Test Error: {}'.format(test_error))