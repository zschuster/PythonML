# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:28:49 2018

@author: ztsch
"""

# Example of nonlinear classification data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c = 'b', marker = 'x', label = '1')
plt.scatter(X_xor[y_xor == -1, 0],X_xor[y_xor == -1, 1],
            c = 'r', marker = 's', label = '-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

# fit kernel SVM with RBF
svm = SVC(kernel = 'rbf', random_state = 0, gamma = .1, C = 10)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.show()

# apply kernelSVM to iris dataset
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

# combine train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


svm = SVC(kernel = 'rbf', random_state = 0, gamma = .2, C = 1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier = svm,
                      test_idx=range(105, 150))
plt.legend(loc = 'upper left')
plt.show()

# what happens when we increase gamma?
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 100, C = 1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier = svm,
                      test_idx=range(105, 150))
plt.legend(loc = 'upper left')
plt.show()

