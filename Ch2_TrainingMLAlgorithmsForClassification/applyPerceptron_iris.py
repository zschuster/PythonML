# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:08:06 2018

@author: ztsch
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simplePerceptron import Perceptron



iris = pd.read_csv('Iris.csv')

y = iris.iloc[0:100, (len(iris.columns) -1)].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = iris.iloc[0:100, [1, 3]].values

plt.scatter(X[:50, 0], X[:50, 1],
            color = 'red', marker = 'o', label = 'setosa')

plt.scatter(X[50:, 0], X[50:, 1],
            color = 'blue', marker = 'x', label = 'versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.show()

# apply perceptron

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

# plot number of misclassifications per epoch
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('# of misclassifications')
plt.show()
