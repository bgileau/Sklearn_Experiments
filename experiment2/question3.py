import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.neighbors import radius_neighbors_graph
import os

import sklearn.utils.graph_shortest_path
from scipy import io

import scipy
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import networkx

print("Part A")

# Create paths to subjects
os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/isomap.mat")

# Demo code:
isomap_dict = scipy.io.loadmat(os_dir_data)  #piazza suggestion
arr = isomap_dict["images"]

arr_list = list(arr)

#print(arr.shape)

epsilon = 12
A = sklearn.neighbors.radius_neighbors_graph(arr.T, radius=epsilon, mode="distance")  # TA suggestion
A = A.todense(order="F")

#print(A.shape)

# Piazza suggestion
graph = networkx.from_numpy_matrix(A)
# Establish layout and set scale for clarity purposes
network_layout = networkx.spring_layout(graph, seed=0, scale=5)
# Draw network and adjust node size/width to declutter a little
networkx.draw_networkx(graph, network_layout, node_size=5, width = .5, with_labels=False, alpha=.8)

gca_plot = plt.gca()

for i in range(0, A.shape[0], 20):
    img_i_arr = arr[:,i].reshape(64,64).T
    img_i = OffsetImage(img_i_arr, zoom=.3, cmap="Greys")
    ab_i = AnnotationBbox(img_i, (network_layout[i][0], network_layout[i][1]), xycoords='data', frameon=False)
    gca_plot.add_artist(ab_i)

plt.show()

################## b:

print("Part B")

# Calculate shortest distance:
D = sklearn.utils.graph_shortest_path.graph_shortest_path(A, directed=False)  # Piazza suggestion
#print(D.shape)

# Extract reduced representation

one_vector = np.ones(A.shape[0])
I = np.diag(np.ones(A.shape[0]))
H = I - (1/A.shape[0]) * np.dot(one_vector, one_vector.T)

#print(H)

# Calculate G
G = (-.5) * H @ D**2 @ H  # D**2 looks better

#print(G.shape)

k = 2
S1,W1 = np.linalg.eig(G)
#print(W1.shape, S1.shape)
#print(f"Largest eigenval: {np.max(S1)}")
idx_eigVal = S1.argsort()[::-1]
S1 = S1[idx_eigVal]
S1 = S1[:k]
W1 = W1[:,idx_eigVal]
W1 = W1[:,:k]

#print(S1)

#print(f"Largest eigenval: {np.max(S1)}")

#print(S1.shape, W1.shape, W1.T.shape, np.diag(S1).shape)

S1_diag = np.diag(S1)
S1_diag = np.sqrt(S1_diag)

Z = (np.dot(W1, S1_diag)).T

#print(Z.shape)

# Create the scatter
plt.scatter(Z[0], Z[1])

# append images to it, but take into account the new indexing.
gca_plot = plt.gca()

for i in range(0, A.shape[0], 10):
    img_i_arr = arr[:,i].reshape(64,64).T
    img_i = OffsetImage(img_i_arr, zoom=.3, cmap="Greys")
    ab_i = AnnotationBbox(img_i, (Z[0][i], Z[1][i]), xycoords='data', frameon=False)
    gca_plot.add_artist(ab_i)

plt.show()

################ C:

print("Part C now")

#A = sklearn.neighbors.radius_neighbors_graph(arr.T, radius=epsilon, mode="distance", metric="manhattan")  # TA suggestion
A = radius_neighbors_graph(arr.T, radius=500, mode="distance", metric="manhattan")
A = A.todense(order="F")

#print(A)

# Calculate shortest distance:
D = sklearn.utils.graph_shortest_path.graph_shortest_path(A, directed=False)  # Piazza suggestion

# Extract reduced representation

one_vector = np.ones(A.shape[0])
I = np.diag(np.ones(A.shape[0]))
H = I - (1/A.shape[0]) * np.dot(one_vector, one_vector.T)

#print(H)

# Calculate G
G = (-.5) * H @ D**2 @ H  # D**2 looks better

k = 2
S1,W1 = np.linalg.eig(G)
#print(W1.shape, S1.shape)
#print(f"Largest eigenval: {np.max(S1)}")
idx_eigVal = S1.argsort()[::-1]
S1 = S1[idx_eigVal]
S1 = S1[:k]
W1 = W1[:,idx_eigVal]
W1 = W1[:,:k]

#print(S1)

#print(f"Largest eigenval: {np.max(S1)}")

#print(S1.shape, W1.shape, W1.T.shape, np.diag(S1).shape)

S1_diag = np.diag(S1)
S1_diag = np.sqrt(S1_diag)

Z = (np.dot(W1, S1_diag)).T

#print(Z.shape)

# Create the scatter
plt.scatter(Z[0], Z[1])

# append images to it, but take into account the new indexing.
gca_plot = plt.gca()

for i in range(0, A.shape[0], 10):
    img_i_arr = arr[:,i].reshape(64,64).T
    img_i = OffsetImage(img_i_arr, zoom=.3, cmap="Greys")
    ab_i = AnnotationBbox(img_i, (Z[0][i], Z[1][i]), xycoords='data', frameon=False)
    gca_plot.add_artist(ab_i)

plt.show()

################# D:
print("Part D, PCA")
# Code from question2.py

######start demo code from question 2 (modified)
Anew1 = arr
m1,n1 = arr.shape

#print(m1,n1)

# PCA

# better residuals
mu1 = np.mean(Anew1,axis = 1)
xc1 = Anew1 - mu1[:,None]
#print(mu1.shape, xc1.shape)
# # Worse residuals
# mu1 = np.mean(Anew1,axis = 0)
# mu1 = np.atleast_2d(mu1)

# xc1 = Anew1 - mu1
# mu1=mu1.T

C1 = np.dot(xc1,xc1.T)/n1

#print(f"C1 shape", C1.shape)

K = 2
print("Eigendecomposition... please wait")
S1,W1 = np.linalg.eig(C1)
#print(W1.shape, S1.shape)
#print(f"Largest eigenval: {np.max(S1)}")
idx_eigVal = S1.argsort()[::-1]
S1 = S1[idx_eigVal]
S1 = S1[:k]
W1 = W1[:,idx_eigVal]
W1 = W1[:,:k]
S1 = S1.real
W1 = W1.real

W1 = W1.reshape(W1.shape[0],k)

#print(W1[:,0].shape, xc1.shape)

dim1 = np.dot(W1[:,0].T,xc1)/math.sqrt(S1[0]) # extract 1st eigenvalues
dim2 = np.dot(W1[:,1].T,xc1)/math.sqrt(S1[1]) # extract 2nd eigenvalue

######end demo code from question 2 (modified)

#print(dim1.shape, dim2.shape)

# Create the scatter
plt.scatter(dim1, dim2)

# append images to it, but take into account the new indexing.
gca_plot = plt.gca()

for i in range(0, A.shape[0], 15):
    img_i_arr = arr[:,i].reshape(64,64).T
    img_i = OffsetImage(img_i_arr, zoom=.3, cmap="Greys")
    ab_i = AnnotationBbox(img_i, (dim1[i], dim2[i]), xycoords='data', frameon=False)
    gca_plot.add_artist(ab_i)

plt.show()