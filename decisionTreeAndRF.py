# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:10:55 2018

@author: ztsch
"""

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


tree = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                              random_state=0)
tree.fit(X_train, y_train)

plot_decision_regions(X = X_combined,
                      y = y_combined,
                      classifier = tree,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

# fit random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators = 1000, # more than the book
                                random_state=1, n_jobs=8) # use 8 cores
forest.fit(X_train, y_train)

plot_decision_regions(X = X_combined,
                      y = y_combined,
                      classifier = forest,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()


