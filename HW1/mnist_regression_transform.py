#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:17:18 2020

@author: nick
"""
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0
    
np.random.seed(42)

def train(X,Y,reg_lambda,p,sigmaSq):
    """
    Train a transformed ridge regression model using closed form solution.

    Parameters
    ----------
    X : n x d numpy array
        Training features
    Y : n x k numpy array
        Training labels
    lambda : float
        Regularization parameter.
    p : positive integer
        The number of rows in the random matrix G.
    sigmaSq : positive float
        The variance of the entries of G.

    Returns
    -------
    W_hat : d x k numpy array
        The prediction matrix.

    """
    
    #Create one-hot encoding for Y
    _Y = np.array([labels_train==i for i in range(0,10)]).astype(int).T
    
    #Generate random transformations
    G = np.random.normal(0,sigmaSq,size=(p,X.shape[-1]))
    b = np.random.uniform(0,2*np.pi,size=(1,p))
    
    #Tranform the features
    _X = np.cos(X @ G.T + b)
    
    #Solve the ridge regression using closed form solution
    A = _X.T @ _X + reg_lambda * np.eye(_X.shape[-1])
    B = _X.T @ _Y
    W_hat = np.linalg.solve(A,B)
    
    return W_hat,G,b

def predict(W,X,p,sigmaSq,G,b):
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
    #Tranform the features
    _X = np.cos(X @ G.T + b)
    
    result = W.T @ _X.T
    
    #Get the index of the largest entry as the class
    Y = np.argmax(result,axis=0)
    
    return Y

if __name__=='__main__':
    
    #Split the training data randomly into training, validation sets
    validationInd = np.random.choice(range(0,X_train.shape[0]),
                                     size=int(.2*X_train.shape[0]),
                                     replace=False)
    X_validation = X_train[validationInd]
    labels_validation = labels_train[validationInd]
    
    #Get leftover indices to be training data
    trainInd = set(range(0,X_train.shape[0]))-set(validationInd)
    trainInd = np.array(list(trainInd))
    X_train = X_train[trainInd]
    labels_train = labels_train[trainInd]
    
    p_min = 10
    p_max = 5000
    p_step = 10
    p_range = list(range(p_min,p_max+1,p_step))
    sigmaSq = .1
    
    #Create arrays to store results from each model
    training_errors = np.zeros(shape=(len(p_range),1))
    validation_errors = np.zeros(shape=(len(p_range),1))
    WList = []
        
    for i,p in enumerate(p_range):
        
        #Solve ridge regression on MNIST dataset
        W_hat,G,b = train(X_train,labels_train,10**-4,p,sigmaSq)
        
        #Predict the labels for each digit
        y_train = predict(W_hat,X_train,p,sigmaSq,G,b)
        y_validation = predict(W_hat,X_validation,p,sigmaSq,G,b)
        
        #Compute the training and test error
        training_error = np.mean(y_train!=labels_train)
        validation_error = np.mean(y_validation!=labels_validation)
        
        training_errors[i] = training_error
        validation_errors[i] = validation_error
        WList.append(W_hat)
        
        print('Done with p={}'.format(p))
    
    # Plot training and validation errors
    fig,ax = plt.subplots(1)
    ax.plot(p_range,100*training_errors.squeeze(),label='Training Error (%)')
    ax.plot(p_range,100*validation_errors.squeeze(),alpha=.75,label='Validation Error (%)')
    ax.set_xlabel('p')
    ax.set_ylabel('Classification Error')
    ax.legend()
    
    # Compute the test error for part b.
    # Since we observed that validation error decreasing monotonically for p<=5000,
    # choose p=5000
    p_hat = 5000
    y_test = predict(W_hat, X_test, p_hat, sigmaSq, G, b)
    test_error = np.mean(y_test!=labels_test)
    
    # compute 95% confidence interval
    delta = .05
    # m is the sample size for computing the average test error
    m = X_test.shape[0]
    ci_hw = np.sqrt(np.log(2/delta)/2/m)
    print('{}% CI (in %): [{:.3f}%,{:.3f}%]'.format((1-delta)*100,(test_error-ci_hw)*100,(test_error+ci_hw)*100))
