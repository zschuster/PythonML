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
# standardize data
SC = StandardScaler()
X_train_std = SC.fit_transform(X_train)
X_test_Std = SC.transform(X_test)

import numpy as np

#compute covarariance matrix, eigenvalues, eigenvectors
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print("\nEigenvalues \n{}".format(eigen_vals))

# plot variance explained ratio fo eigenvalues
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]

cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1, 14), var_exp, alpha = 0.5, align = 'center',
        label = 'individual explained variance')
plt.step(range(1, 14), cum_var_exp, where = 'mid',
         label = 'cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.show()


""" Feature Transformation """

# sort eigen pairs in descending order

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse = True)

# take two largest eigen pairs for illustration

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))

print("Matrix W:\n", w)

# use projection matrix to get lower dimensional tranformed data
X_train_std[0, :].dot(w) # example with first row of training data

X_train_pca = X_train_std.dot(w)

# plot the newly transformed data

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label = l, marker = m)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc = 'lower left')
plt.show()


""" Show decision regions for a LR trained on pca tranformed data """

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from plot_decision_regions import plot_decision_regions

pca = PCA(n_components = 2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_Std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()



