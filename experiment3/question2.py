import csv
import numpy as np
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import os

import seaborn

from scipy.stats import multivariate_normal as mvn

# Create paths to subjects
os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

n90pol_df = pd.read_csv(os_dir_data + "n90pol.csv")

amygdala_arr = n90pol_df["amygdala"]
acc_arr = n90pol_df["acc"]

amygdala_arr_np = np.array(amygdala_arr)
acc_arr_np = np.array(acc_arr)
# ############## (a)
print("part a")

print("histograms")
# make histograms:
print("amygdala")
plt.hist(amygdala_arr, bins=12)  # Piazza suggestion
plt.show()

print("acc")
plt.hist(acc_arr, bins=12)  # Piazza suggestion
plt.show()

# make KDEs:
print("KDEs")
h = .2
print("amygdala")
amygdala_arr.plot.kde(bw_method=h)  # Piazza suggestion
plt.show()

print("acc")
acc_arr.plot.kde(bw_method=h)  # Piazza suggestion
plt.show()

# ############### (b)
print("part b")
# Demo code start:
nbin = 12
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(amygdala_arr_np, acc_arr_np, bins=nbin)
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)
dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz )

# Demo code end
plt.show()

################# (c)
print("part c")
h = .32
seaborn.kdeplot(x=amygdala_arr_np, y=acc_arr_np,bw=h, shade=True)

plt.show()

########## (d) + (e)
unique_orientations = [2,3,4,5]
################# (d)
print("part d")

sample_mean_dict = {2: {"amygdala":None, "acc":None},
                    3: {"amygdala":None, "acc":None},
                    4: {"amygdala":None, "acc":None},
                    5: {"amygdala":None, "acc":None}}

for orientation in unique_orientations:
    orientation_df = n90pol_df[n90pol_df["orientation"] == orientation]
    print(f"oritentation: {orientation}")

    # make KDEs:
    print("Amygdala KDE")
    h = .3
    orientation_df["amygdala"].plot.kde(bw_method=h)  # Piazza suggestion
    plt.show()

    print("ACC KDE")
    orientation_df["acc"].plot.kde(bw_method=h)  # Piazza suggestion
    plt.show()

    sample_mean_dict[orientation]["amygdala"] = round(orientation_df["amygdala"].mean(),10)
    sample_mean_dict[orientation]["acc"] = round(orientation_df["acc"].mean(),10)

    #print(orientation_df)

print(sample_mean_dict)

###################### (e)
print("part e")

for orientation in unique_orientations:
    orientation_df = n90pol_df[n90pol_df["orientation"] == orientation]
    print(f"oritentation: {orientation}")

    # make KDEs:
    h = .32
    seaborn.kdeplot(x=orientation_df["amygdala"], y=orientation_df["acc"],bw=h, shade=True)

    plt.show()


    