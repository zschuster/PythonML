# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:31:25 2018

@author: ztsch
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

# load in iris data set
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,
                                                    random_state = 0)

# scale covariates
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# fit support vector machine
svm = SVC(kernel = 'linear', C = 1, random_state=0)
svm.fit(X_train_std, y_train)

# plot svm decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = svm,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

