'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
    
    Modified by Nick Terry for CSE 546 HW1 A.4
'''

import numpy as np
import linreg_closedform as lrc

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.degree = degree
        self.reg_lambda = reg_lambda
        
        self.stand = None
        
        self.linreg = lrc.LinearRegressionClosedForm(self.reg_lambda)

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        feature_mat = np.array([X**d for d in range(1,degree+1)])
        
        if len(feature_mat.shape) > 2:
            feature_mat = np.reshape(feature_mat,(self.degree,feature_mat.shape[1]))
        
        feature_mat = feature_mat.T
        
        return feature_mat

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        feature_mat = self.polyfeatures(X, self.degree)
        feature_mat = self.standardize(feature_mat)
        
        self.linreg.fit(feature_mat, y)
        

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        feature_mat = self.polyfeatures(X, self.degree)
        feature_mat = self.standardize(feature_mat)
        
        yhat = self.linreg.predict(feature_mat)
        
        return yhat
        
    def standardize(self, feature_mat):
        """
        Standardize the features by scaling training features to [0,1].
        Scaling factors are saved for later.

        Parameters
        ----------
        feature_mat : n x d np array
            The feature matrix to be scaled.

        Returns
        -------
        feature_mat : n x d np array
            The rescaled feature matrix

        """      
        if self.stand is None:
            
            #Compute the max for each feature for scaling
            self.stand = np.max(np.abs(feature_mat),axis=0).reshape(
                (1,feature_mat.shape[1]))
        
            zeroInds = np.where(self.stand==0)
            self.stand[zeroInds] = 1
            
        feature_mat = feature_mat/np.repeat(self.stand,
                                            feature_mat.shape[0],
                                            axis=0)
        return feature_mat

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(1,n+1):
        polyreg = PolynomialRegression(degree,reg_lambda)
        polyreg.fit(Xtrain[:i,:], Ytrain[:i])
        
        errorTrain[i-1] = np.mean((polyreg.predict(Xtrain[:i,:])-Ytrain[:i])**2)
        errorTest[i-1] = np.mean((polyreg.predict(Xtest)-Ytest)**2)

    return errorTrain, errorTest
