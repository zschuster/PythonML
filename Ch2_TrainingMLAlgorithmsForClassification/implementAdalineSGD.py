# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 09:04:29 2018

@author: ztsch
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from AdalineSGD import AdalineSGD



iris = pd.read_csv('Iris.csv')

y = iris.iloc[0:100, (len(iris.columns) -1)].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = iris.iloc[0:100, [1, 3]].values

# standardize covariates (only works for 2 column array)
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(n_iter = 15, eta = .01, random_state = 1)
ada.fit(X_std, y)

# plot cost against iteration
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Avg Cost')
plt.show