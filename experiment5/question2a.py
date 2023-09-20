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

np.random.seed(2)  # piazza suggestion

os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

real_estate_df = pd.read_csv(os_dir_data + "RealEstate.csv", header=0)
real_estate_df.drop(columns=["MLS", "Location"], inplace=True)

########### (a):

# Shuffle data (according to TA with seed=2)
real_estate_df = real_estate_df.sample(frac=1, random_state=2)  # piazza suggestion

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



#print(real_estate_df)

# Shuffle data (according to TA with seed=2)
real_estate_df = real_estate_df.sample(frac=1, random_state=2)  # piazza suggestion

# Show CV Curve
ridge = Ridge(random_state=2, normalize=False, fit_intercept=False)  
params = {"alpha":np.arange(1,81)}

neg_scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better=False)

nonprice_df = real_estate_df.drop(columns=["Price"])
x_arr = nonprice_df.values
y_arr = real_estate_df["Price"].values

x_arr, y_arr = shuffle(x_arr, y_arr, random_state=2)  # works??

alpha_mse_dict = {}
for i in range(1, 81):
    ridge = Ridge(random_state=2, alpha=i)  # gives alpha = 2, but bad values?  , normalize=False, fit_intercept=False
    scores = cross_val_score(ridge, x_arr, y_arr, scoring=neg_scorer)  # 5 fold cross val built-in.
    alpha_mse_dict[i] = abs(np.average(scores))

print(alpha_mse_dict)
min_val = min(alpha_mse_dict.values())
print(f"min val: {min_val}")
alpha_val = list(alpha_mse_dict.values()).index(min_val) + 1
#print("index of min val: ",)
print(f"ALPHA value for min value: {alpha_val}")

# CV 
plt.scatter(alpha_mse_dict.keys(),alpha_mse_dict.values())
plt.show()

ridge_optimal = Ridge(random_state=2, alpha=alpha_val)  # 2 is the optimal alpha found above

clf = ridge_optimal.fit(x_arr, y_arr)
print(f"Coefficients for fitted model at alpha {alpha_val}")
print(nonprice_df.columns)
print(clf.coef_)

# piazza
rss = min_val * y_arr.shape[0]
print(f"RSS = abs(cross val minimum val) * data size = {min_val} * {y_arr.shape[0]} = {rss}")  