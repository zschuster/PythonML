# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 09:32:28 2018

@author: ztsch
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
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

# train a perceptron model, which uses One-vs.-Rest method for multi-classification
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# make predictions on the test set
y_pred = ppn.predict(X_test_std)

print('Missclassified samples: {}'.format((y_test != y_pred).sum()))

# get accuracy
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

# plot decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = ppn,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

