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

from sklearn.linear_model import Ridge, Lasso, lasso_path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, metrics, decomposition, model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn import metrics

import random

import seaborn
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle

np.random.seed(3)  # piazza suggestion

os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

real_estate_df = pd.read_csv(os_dir_data + "RealEstate.csv", header=0)
real_estate_df.drop(columns=["MLS", "Location"], inplace=True)

# Shuffle data (according to TA with seed=2)
real_estate_df = real_estate_df.sample(frac=1, random_state=3)  # piazza suggestion

# print(real_estate_df)

# one-hot-keying Status variable
status_arr = real_estate_df["Status"].values
hot_encoder = OneHotEncoder()
val = hot_encoder.fit_transform(status_arr.reshape(-1,1)).toarray()
real_estate_df.drop(columns=["Status"], inplace=True)
real_estate_df["Status_Foreclosure"] = val[:,0]
real_estate_df["Status_Regular"] = val[:,1]
real_estate_df["Status_Short Sale"] = val[:,2]

# print(real_estate_df)



# Scale indep vars (not price)
scaler = StandardScaler()
# Convert to numpy arrs and then reshape for fit_transform
bedrooms_arr = real_estate_df["Bedrooms"].values
bathrooms_arr = real_estate_df["Bathrooms"].values
size_arr = real_estate_df["Size"].values
pricesq_arr = real_estate_df["Price/SQ.Ft"].values

real_estate_df["Bedrooms"] = scaler.fit_transform(bedrooms_arr.reshape(-1,1))
real_estate_df["Bathrooms"] = scaler.fit_transform(bathrooms_arr.reshape(-1,1))
real_estate_df["Size"] = scaler.fit_transform(size_arr.reshape(-1,1))
real_estate_df["Price/SQ.Ft"] = scaler.fit_transform(pricesq_arr.reshape(-1,1))

# Shuffle data (according to TA with seed=2)
real_estate_df = real_estate_df.sample(frac=1, random_state=3)  # piazza suggestion

neg_scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better=False)

nonprice_df = real_estate_df.drop(columns=["Price"])
x_arr = nonprice_df.values
y_arr = real_estate_df["Price"].values

x_arr, y_arr = shuffle(x_arr, y_arr, random_state=3)  # works??

############# (b):

x_lasso_arr, y_lasso_arr = shuffle(x_arr, y_arr, random_state=3)  # works??

print("Working on LASSO CV")
alpha_mse_dict = {}
laso_coef_dict = {1:[],2:[],3:[],4:[],5:[], 6:[], 7:[]}
for i in range(1, 3001):  #3001
    lasso = Lasso(random_state=3, alpha=i)  # gives alpha = 2, but bad values?  , normalize=False, fit_intercept=False
    lasso.fit(x_lasso_arr, y_lasso_arr)
    coefficients_arr = np.array(lasso.coef_)
    laso_coef_dict[1].append(coefficients_arr[0])
    laso_coef_dict[2].append(coefficients_arr[1])
    laso_coef_dict[3].append(coefficients_arr[2])
    laso_coef_dict[4].append(coefficients_arr[3])
    laso_coef_dict[5].append(coefficients_arr[4])
    laso_coef_dict[6].append(coefficients_arr[5])
    laso_coef_dict[7].append(coefficients_arr[6])
    
    scores = cross_val_score(lasso, x_lasso_arr, y_lasso_arr, scoring=neg_scorer)  # 5 fold cross val built-in.

    alpha_mse_dict[i] = abs(np.average(scores))

#print(alpha_mse_dict)
min_val = min(alpha_mse_dict.values())
print("min val", min_val)
alpha_val = list(alpha_mse_dict.values()).index(min_val) + 1
#print("index of min val: ",)
print(f"ALPHA value for min value: {alpha_val}")

labels = ['Bedrooms', 'Bathrooms', 'Size', 'Price/SQ.Ft', 'Status_Foreclosure', 'Status_Regular', 'Status_Short Sale']

# LASSO PATH
for x, label in zip(range(0, 7), labels):
    plt.plot(np.log(np.arange(1,3001)),laso_coef_dict[x+1], label=label)  # piazza suggestion to log alphas.
    
plt.legend()
plt.xlabel("log(alpha)")
plt.ylabel("coefficient vals")
plt.show()

# CV
plt.scatter(alpha_mse_dict.keys(),alpha_mse_dict.values())
plt.show()

lasso_optimal = Lasso(random_state=3, alpha=alpha_val)  # 2 is the optimal alpha found above

clf = lasso_optimal.fit(x_arr, y_arr)
print(f"Coefficients for fitted model at alpha {alpha_val}")
print(clf.coef_)