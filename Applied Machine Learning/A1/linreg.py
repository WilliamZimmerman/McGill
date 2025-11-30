import numpy as np
import pandas as pd

class LinReg:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        print("INIT DONE")
        pass

    def fit_regular(self, X ,y):
        X = X.values
        y=y.values
    
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        XT = X.T
        XTX = np.matmul(XT, X)  # X^T X
        XTX_inv = np.linalg.inv(XTX)  # (X^T X)^-1
        beta = np.matmul(XTX_inv, np.matmul(XT, y))  # beta = (X^T X)^-1 X^T y
        return beta

    '''
    When using least sqaures the Gramian Matrix XTX is very poorly conditioned.
    Using QR can help increase the stability as well as the 
    '''
    def fit_QR(self, X, y):
        X = X.values
        y=y.values
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        Q,R = np.linalg.qr(X)
        beta = np.linalg.solve(R, np.dot(Q.T, y))
        return beta
    
    #def fit_stochastic(self, X,y,batch_size,iter,learning_rate):



    def predict(self, X, beta):
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        y_hat = X @ beta
        return y_hat
    

    
    
    
