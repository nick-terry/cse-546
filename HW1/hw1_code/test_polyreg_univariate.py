"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, [0]]
    y = allData[:, [1]]

    for reg_lambda in [0,1e-6,1e-5,1e-3]:
        # regression with degree = d
        d = 8
        model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
        model.fit(X, y)
    
        # output predictions
        xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
        ypoints = model.predict(xpoints)
    
        # plot curve
        plt.figure()
        plt.plot(X, y, 'rx')
        plt.title('PolyRegression with d = {},reg_lambda={}'.format(d,reg_lambda))
        plt.plot(xpoints, ypoints, 'b-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
