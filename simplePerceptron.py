# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:49:04 2018

@author: ztsch
"""

import numpy as np

class Perceptron(object):
    """Perceptron Classifier.
    
    Params
    -------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        # of passes over the training data set
        
    Attributes
    -------
    w_ : 1d-array
        weights after fitting
        
    errors_ : list
        Number of misclassifications in every epoch
    """
    
    def __init__(self, eta = 0.01, n_iter = 10) :
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y) :
        """Fit training data.
        
        Params
        -----
        X: {array like}, shape = [n_samples, n_features].
        training vectors, where n_samples is the number
        of samples and n_feautures is the number of features
        
        y: array like, shape = [n_samples]
            target vales. (response)
            
        Returns
        -------
        
        self: object
        """
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target * self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return(self)
        
        def net_input(self, X):
            """Calculate net input"""
            
            return(np.dot(X, self.w_[1:]) + self.w_[0])
            
        def predict(self, X):
            """Return class label after unit step"""
            
            return(np.where(self.net_input(X) >= 0.0, 1, -1))
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    