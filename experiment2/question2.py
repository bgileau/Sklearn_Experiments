import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import os

from skimage import io
from skimage.transform import downscale_local_mean


def read_and_downsample_image(image_dir):
    image_normal = io.imread(image_dir, as_gray=True)
    image_ds = downscale_local_mean(image=image_normal, factors=(4,4)).flatten()
    return image_ds
    

# Create paths to subjects
os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/yalefaces/")

data_files = os.listdir(os_dir_data)

test1_image_arr = []
test2_image_arr = []
eigenface1_image_arr = []
eigenface2_image_arr = []
for files in data_files:
    if "subject01" in files:
        if "-test" in files:
            test1_image_arr.append(read_and_downsample_image(os_dir_data + files))
        else:
            eigenface1_image_arr.append(read_and_downsample_image(os_dir_data + files))
    elif "subject02" in files:
        if "-test" in files:
            test2_image_arr.append(read_and_downsample_image(os_dir_data + files))
        else:
            eigenface2_image_arr.append(read_and_downsample_image(os_dir_data + files))

eigenface1_image_arr = np.array(eigenface1_image_arr)
eigenface2_image_arr = np.array(eigenface2_image_arr)

eigenfaces_image_arr = [eigenface1_image_arr, eigenface2_image_arr]

#print(eigenface1_image_arr.shape, eigenface2_image_arr.shape)

top_eig_face_dict = {"subject01": None, "subject02": None}

for i in range(0, len(eigenfaces_image_arr)):
    if i == 0:
        sub_name = "subject01"
    else:
        sub_name = "subject02"
    ########## Start demo code block:  (modified)
    Anew1 = eigenfaces_image_arr[i]
    m1,n1 = eigenfaces_image_arr[i].shape

    # PCA

    # better residuals
    mu1 = np.mean(Anew1,axis = 1)
    xc1 = Anew1 - mu1[:,None]

    # # Worse residuals
    # mu1 = np.mean(Anew1,axis = 0)
    # mu1 = np.atleast_2d(mu1)

    # xc1 = Anew1 - mu1
    # mu1=mu1.T

    C1 = np.dot(xc1,xc1.T)/m1

    K = 6
    S1,W1 = ll.eigs(C1,k = K)
    S1 = S1.real
    W1 = W1.real

    dim1 = np.dot(W1[:,0].T,xc1)/math.sqrt(S1[0]) # extract 1st eigenvalues
    dim2 = np.dot(W1[:,1].T,xc1)/math.sqrt(S1[1]) # extract 2nd eigenvalue
    dim3 = np.dot(W1[:,2].T,xc1)/math.sqrt(S1[2])
    dim4 = np.dot(W1[:,3].T,xc1)/math.sqrt(S1[3])
    dim5 = np.dot(W1[:,4].T,xc1)/math.sqrt(S1[4])
    dim6 = np.dot(W1[:,5].T,xc1)/math.sqrt(S1[5])

    ########## End demo code block  (modified)

    top_eig_face_dict[sub_name] = dim2

    #plt.imshow()

    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc1.gif"),dim1.reshape(61,80), cmap="gray")
    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc2.gif"),dim2.reshape(61,80), cmap="gray")
    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc3.gif"),dim3.reshape(61,80), cmap="gray")
    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc4.gif"),dim4.reshape(61,80), cmap="gray")
    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc5.gif"),dim5.reshape(61,80), cmap="gray")
    plt.imsave(os.path.join(os_dir, f"output/{sub_name}-pc6.gif"),dim6.reshape(61,80), cmap="gray")

########## (2.b)
test1_image_arr = np.array(test1_image_arr)
test2_image_arr = np.array(test2_image_arr)

# Center test data
print(test1_image_arr.shape, top_eig_face_dict["subject01"].shape)

# np doc recommendeation to create column vector
top_eig_face_dict["subject01"] = np.atleast_2d(top_eig_face_dict["subject01"]).T  
top_eig_face_dict["subject02"] = np.atleast_2d(top_eig_face_dict["subject02"]).T

# create column vector
test1_image_arr = test1_image_arr.reshape(test1_image_arr.shape[1],1)
test2_image_arr = test2_image_arr.reshape(test2_image_arr.shape[1],1)

print(test1_image_arr.shape, top_eig_face_dict["subject01"].shape, top_eig_face_dict["subject02"].shape)

sub1_mu = np.mean(top_eig_face_dict["subject01"],axis = 0)

test1_image_arr_centered = test1_image_arr - sub1_mu
test2_image_arr_centered = test2_image_arr - sub1_mu

#print(test1_image_arr_centered.shape, test2_image_arr_centered.shape)

# Solved residual formula
e1 = np.dot(top_eig_face_dict["subject01"],top_eig_face_dict["subject01"].T)
e2 = np.dot(e1, test1_image_arr_centered)
expression11 = test1_image_arr_centered - e2
s11 = np.linalg.norm(expression11) ** 2

expression12 = test2_image_arr_centered - np.dot(np.dot(top_eig_face_dict["subject01"],top_eig_face_dict["subject01"].T), test2_image_arr_centered)
s12 = np.linalg.norm(expression12) ** 2

sub2_mu = np.mean(top_eig_face_dict["subject02"],axis = 0)

test1_image_arr_centered = test1_image_arr - sub2_mu
test2_image_arr_centered = test2_image_arr - sub2_mu

expression21 = test1_image_arr_centered - np.dot(np.dot(top_eig_face_dict["subject02"],top_eig_face_dict["subject02"].T), test1_image_arr_centered)
s21 = np.linalg.norm(expression21) ** 2

expression22 = test2_image_arr_centered - np.dot(np.dot(top_eig_face_dict["subject02"],top_eig_face_dict["subject02"].T), test2_image_arr_centered)
s22 = np.linalg.norm(expression22) ** 2

print(s11, s12, s21, s22)
