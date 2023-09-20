import csv
import numpy as np
from numpy.core.fromnumeric import argmax
import numpy.matlib
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import os
import scipy.io
import math
import seaborn

from sklearn import cluster

from PIL import Image as im

from scipy.stats import multivariate_normal as mvn

# Create paths to subjects
os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

############## (b)
data = scipy.io.loadmat(os_dir_data + "data.mat")
label = scipy.io.loadmat(os_dir_data + "label.mat")

# demo code:

y = label["trueLabel"].T
data = data["data"].T



#################### Demo code start
ndata = data

m, n = ndata.shape
C = np.matmul(ndata.T, ndata)/m

print(f"C shape: {C.shape}")

# pca the data
d = 4  # reduced dimension
V,S,_ = np.linalg.svd(C)
V = V[:, :d]
S = S[:d]  # grab the top 4 eigenvalues as well

print(f"v shape {V.shape}")

# project the data to the top 4 principal directions
pdata = np.dot(ndata,V)

print(f"pdata shape: {pdata.shape}")

maxIter= 100
tol = 1e-3
K = 2

# random seed
seed = 90  # 89 

pi = np.random.random(K)
pi = pi/np.sum(pi)

print(f"pi shape {pi.shape}")

mu = np.random.randn(K,4)
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # to ensure the covariance psd
    # np.random.seed(seed)
    dummy = np.random.randn(4, 4)
    sigma.append(dummy@dummy.T + np.eye(4,4))

#plt.ion()

# initialize the posterior
tau = np.full((m, K), fill_value=0.)

log_likelihood_arr = []
for ii in range(maxIter):

    # E-step
    for kk in range(K):
        likelihood_kk = mvn.pdf(pdata, mu[kk], sigma[kk])
        tau[:, kk] = pi[kk] * likelihood_kk
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))

    #print(sum_tau)
    
    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk])/m
        
        # update component mean
        mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)
        
        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m,1)) # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)

    #################### Demo code end (modified)
    # Log-likelihood work
    val_arr = []
    for kk in range(K):
        val_arr.append(mvn.pdf(pdata, mu[kk], sigma[kk]))

    val_arr = np.array(val_arr).T
    val = val_arr.sum(axis=1)
    log_val = np.log(val)
    
    val_log_sum = log_val.sum()
    log_likelihood_arr.append(val_log_sum)

    # Demo code stuff (modified) to know when to stop the algorithm
    print('-----iteration---',ii)
    diff = np.linalg.norm(mu-mu_old)
    if diff < tol:
        print('training coverged')
        break
    else:
        print(f"diff: {diff}")
    mu_old = mu.copy()
    if ii==99:
        print('max iteration reached')
        break


# Plot the log-likelihood
plt.plot(np.arange(0, ii+1),log_likelihood_arr)
plt.show()

######### (c)
print(pdata.shape, pi.shape, tau.shape, mu.shape, V.shape, S.shape, len(sigma))

lambda_val = np.diag(S)

# weights:
print(f"Weights: {pi}")

# Mean of each comp
print(f"Mean of each component: {tau.sum(axis=1)}")

pdata_avg = pdata.sum(axis=1)  # sum across pc

for kk in range(K):
    print(f"Ok K: {kk}")
    mu_tilde_k = (V @ (lambda_val ** .5) @ mu[kk].T) + np.mean(pdata_avg)  # TA Piazza suggestion

    mu_tilde_k_reshaped = mu_tilde_k.reshape(28,28)

    print("Reshaped image")
    mu_tilde_k_reshaped = preprocessing.minmax_scale(mu_tilde_k_reshaped.T, feature_range=(0,255))
    img = im.fromarray(mu_tilde_k_reshaped)
    img.show()

    print("Covariance matrix")
    cov_rescaled = preprocessing.minmax_scale(sigma[kk], feature_range=(0,255))
    seaborn.heatmap(cov_rescaled, cmap="gray")
    plt.show()

##### (d)
estimated_labels = np.argmax(tau, axis=1)

print(estimated_labels.shape, y.shape)

kmeans = cluster.KMeans(n_clusters=2).fit(pdata)
test_df = pd.DataFrame()

test_df["estimated_labels"] = estimated_labels
test_df["true_labels"] = y[:]
test_df["kmeans_labels"] = kmeans.labels_

label_2_df = test_df[test_df["true_labels"] == 2]
label_2_mode_gmm = int(pd.Series(label_2_df["estimated_labels"]).mode())
label_2_mode_kmeans = int(pd.Series(label_2_df["kmeans_labels"]).mode())


estimate_2_gmm = len(label_2_df[label_2_df["estimated_labels"] == label_2_mode_gmm])
mismatch_2_gmm =  1 - (estimate_2_gmm / len(label_2_df["true_labels"]))

estimate_2_kmeans = len(label_2_df[label_2_df["kmeans_labels"] == label_2_mode_kmeans])
mismatch_2_kmeans =  1 - (estimate_2_kmeans / len(label_2_df["true_labels"]))

label_6_df = test_df[test_df["true_labels"] == 6]
label_6_mode_gmm = int(pd.Series(label_6_df["estimated_labels"]).mode())
label_6_mode_kmeans = int(pd.Series(label_6_df["kmeans_labels"]).mode())

estimate_6_gmm = len(label_6_df[label_6_df["estimated_labels"] == label_6_mode_gmm])
mismatch_6_gmm =  1 - (estimate_6_gmm / len(label_6_df["true_labels"]))

estimate_6_kmeans = len(label_6_df[label_6_df["kmeans_labels"] == label_6_mode_kmeans])
mismatch_6_kmeans =  1 - (estimate_6_kmeans / len(label_6_df["true_labels"]))

print("GMM Mismatches: 2,6")
print(mismatch_2_gmm, mismatch_6_gmm)
print("KMeans Mismatches: 2,6")
print(mismatch_2_kmeans, mismatch_6_kmeans)