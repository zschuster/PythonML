# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:53:10 2018

@author: ztsch
"""

import pandas as pd

df_wine = pd.read_csv('wine.data', header = None)

# create training and test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read data frame into array
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,
                                                    random_state = 0)

SC = StandardScaler()
X_train_std = SC.fit_transform(X_train)
X_test_Std = SC.transform(X_test)

import numpy as np

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print("\nEigenvalues \n{}".format(eigen_vals))