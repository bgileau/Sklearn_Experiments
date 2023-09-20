import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import real
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import os

from skimage import io
from scipy.io import loadmat
from sklearn import tree

from sklearn.linear_model import Ridge, Lasso, lasso_path, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, metrics, decomposition, model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import random

import seaborn
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle

os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

cs_data = loadmat(os_dir_data + "cs.mat")

img_arr = cs_data["img"]
print(img_arr.shape)

np.random.seed(2)
random_state_val = 2

# plt.imshow(img_arr.reshape(50,50), cmap="gray")
# plt.show()

# y = Ax + n
x = img_arr.reshape(2500,1)
n = np.random.normal(loc=0,scale=5, size=(1300,1))  # scale is std. dev

A = np.random.normal(loc=0,scale=1, size=(1300,2500))

print(x.shape, n.shape, A.shape, A.dot(x).shape)

y = A.dot(x) + n

print(y.shape)


alpha_upper_limit = 100

########## a
print("Part A")
print("LASSO CV, 10 folds, 100 alphas")
clf_lasso = LassoCV(cv=10, n_alphas=alpha_upper_limit, random_state=random_state_val)
clf_lasso.fit(A, y.ravel())  # fitting with A and y is correct per piazza TA

lasso_path = clf_lasso.mse_path_
lasso_alphas = clf_lasso.alphas_
lasso_best_alpha = clf_lasso.alpha_

lasso_path_avg = np.mean(lasso_path, axis=1)
print(lasso_path_avg.shape)

print(f"Best Alpha: {lasso_best_alpha}")
plt.plot(lasso_alphas, lasso_path_avg)  # np.arange(1,alpha_upper_limit+1)
plt.show()

# Show recovered image with best alpha
clf_lasso2 = Lasso(alpha=lasso_best_alpha, random_state=random_state_val)
clf_lasso2.fit(A, y.ravel())
lasso2_coef = clf_lasso2.coef_

plt.imshow(lasso2_coef.reshape(50,50), cmap="gray")
plt.show()


############ b

print("Part B")

neg_scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better=True)

alpha_mse_dict = {}
ridge_coef_dict = {}
for i in range(1, 101):
    if i % 10 == 0: 
        print(f"On alpha iteration: {i}")
    ridge = Ridge(alpha=i, random_state=random_state_val)
    ridge.fit(A, y.ravel())
    coefficients_arr = np.array(ridge.coef_)
    
    scores = cross_val_score(ridge, A, y.ravel(), scoring=neg_scorer, cv=10)  # 5 fold cross val built-in.

    alpha_mse_dict[i] = abs(np.average(scores))

min_val = min(alpha_mse_dict.values())
print("min val", min_val)
alpha_val = list(alpha_mse_dict.values()).index(min_val) + 1
print(f"ALPHA value for min value: {alpha_val}")

# CV
plt.scatter(alpha_mse_dict.keys(),alpha_mse_dict.values())
plt.show()

ridge_optimal = Ridge(alpha=min_val, random_state=random_state_val)
ridge_optimal.fit(A, y.ravel())

plt.imshow(ridge_optimal.coef_.reshape(50,50), cmap="gray")
plt.show()