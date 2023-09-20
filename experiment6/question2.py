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

from sklearn.linear_model import Ridge, Lasso, lasso_path
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

spam_df = pd.read_csv(os_dir_data + "spambase.data", header=None)

print(spam_df.shape)

#### a+b+c:
random_state_val = 5
spam_df_train, spam_df_test = train_test_split(spam_df, test_size=.2, train_size=.8, random_state=random_state_val)

spam_df_train_X = spam_df_train.iloc[:,:-1]
spam_df_train_Y = spam_df_train.iloc[:,-1]

spam_df_test_X = spam_df_test.iloc[:,:-1]
spam_df_test_Y = spam_df_test.iloc[:,-1]

print(spam_df_train_X.shape, spam_df_train_Y.shape)

########## (a)
print("Part A")
clf_CART = DecisionTreeClassifier()
clf_CART.fit(spam_df_train_X, spam_df_train_Y)  # use 80/20 split for part (a) as well as per Piazza

plt.figure(figsize=(18,8))  # to help fit
tree_plot = plot_tree(clf_CART, max_depth=4, fontsize=6)  # piazza suggestion for max depth
plt.show()

######### (b)
print("Part B")
spam_pred_CART = clf_CART.predict(spam_df_test_X)
accuracy_score_CART = metrics.accuracy_score(y_true=spam_df_test_Y, y_pred=spam_pred_CART)  # piazza suggestion for accuracy_score

# Plot test error vs # of trees
max_tree_count_idx = 201  # this count is "more than sufficient" according to TA on Piazza
test_error_RF_arr = []
test_error_CART_arr = []
for i in range(1, max_tree_count_idx):  # 201
    if i % 20 == 0:  # print every 20 iterations
        print(f"On # trees count: {i}/{max_tree_count_idx-1}")
    clf_forest_i = RandomForestClassifier(n_estimators=i)
    clf_forest_i.fit(spam_df_train_X, spam_df_train_Y)

    spam_pred_RF = clf_forest_i.predict(spam_df_test_X)
    accuracy_score_RF_i = metrics.accuracy_score(y_true=spam_df_test_Y, y_pred=spam_pred_RF)  # piazza suggestion for accuracy_score

    test_error_RF_arr.append(1-accuracy_score_RF_i)
    test_error_CART_arr.append(1-accuracy_score_CART)

plt.plot(np.arange(1,max_tree_count_idx),test_error_RF_arr)
plt.plot(np.arange(1,max_tree_count_idx),test_error_CART_arr)
plt.show()

clf_forest = RandomForestClassifier(n_estimators=max_tree_count_idx)
clf_forest.fit(spam_df_train_X, spam_df_train_Y)

print("Test Errors:")

print(f"Part b (CART) Test Error: {1-accuracy_score_CART}")

# spam_pred_RF = clf_forest.predict(spam_df_test_X)
# accuracy_score_RF = metrics.accuracy_score(y_true=spam_df_test_Y, y_pred=spam_pred_RF)  # piazza suggestion for accuracy_score
test_error_RF_nparr = np.array(test_error_RF_arr)
min_idx = np.argmin(test_error_RF_nparr)
print("(RF)")
print(f"# of trees with minimum error: {min_idx+1}")
print(f"Test Error for {min_idx+1} trees: {test_error_RF_nparr[min_idx]}")


################# (c)

print("Part C")
spam_df_train_nonspam = spam_df_train[spam_df_train.iloc[:,-1] == 0]  # 0 is nonspam
spam_df_train_nonspam_X = spam_df_train_nonspam.iloc[:,:-1]
spam_df_train_nonspam_Y = spam_df_train_nonspam.iloc[:,-1]

print(spam_df_train_nonspam.shape, spam_df_train.shape)

clf_svm = OneClassSVM(kernel="rbf", gamma="auto")  # scale is worse than auto (show in report)
clf_svm.fit(spam_df_train_nonspam_X, spam_df_train_nonspam_Y)

spam_pred_svm = np.array(clf_svm.predict(spam_df_test_X))

# Outlier = spam = -1 according to TA Piazza docs link. Convert 1 to be 0's (nonspam) and -1's to 1's 
spam_pred_svm = np.where(spam_pred_svm == 1, 0, spam_pred_svm)
spam_pred_svm = np.where(spam_pred_svm == -1, 1, spam_pred_svm)


accuracy_score_SVM = metrics.accuracy_score(y_true=spam_df_test_Y, y_pred=spam_pred_svm)
print(f"Part C (SVM) Test Error: {1-accuracy_score_SVM}")

# no plot needed for c
