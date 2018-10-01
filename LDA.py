# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:52:34 2018

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

# We have already standardized the data
# Compute mean vectors 

np.set_printoptions(precision = 4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis = 0))
    print("MV {}: {}\n".format(label, mean_vecs[label -1]))

# compute within class scatter matrices

d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter

print("Within-class scatter matrix: {}".format(S_W.shape))


# We assume classes are uniformly distributed. Is this true?
np.bincount(y_train)[1:]

# This isn't true, so we need to scale the scatter matrix

d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: {}'.format(S_W.shape))


# calculate between-class scatter matrix

mean_overall = np.mean(X_train_std, axis = 0)
d = 13 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print("Between-class scatter matrix: {}".format(S_B.shape))

# perform eigendecomposition on inverse of within times between
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# sort eigen pairs in descending order by eigen values
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) \
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True)

print("Eigenvalues in decreasing order:\n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# plot discriminants
import matplotlib.pyplot as plt
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse = True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha = .5, align = 'center',
        label = 'individual "discriminability"')
plt.step(range(1, 14), cum_discr, where = 'mid',
         label = 'cumulative "discriminability"')
plt.ylim([-.1, 1.1])
plt.legend(loc = 'best')
plt.show()

# create projection matrix
w = np.hstack((((eigen_pairs[0][1][:, np.newaxis].real) * -1),
               eigen_pairs[1][1][:, np.newaxis].real))

print("Matrix W: \n", w)
