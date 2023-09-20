import scipy.spatial as sp
import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
import os
import math
import random
import copy
import time

from scipy.sparse import csc_matrix


def kmeans(pixels, k, distance_type):
    # Step 1: Randomly initialize k cluster centers RGB in [0,255]. Want k rows (b/c # clusters, and 3 cols)
    # (semi) demo code
    c = np.random.randint(low=0, high=256,size=(k,3))
    c_old = copy.deepcopy(c)  # modified demo code
    print(f"Cluster center Initialization: {c}")

    if distance_type == "L2 distance":
        print("Using L2 distance")
        ord_arg = 2
        cdist_arg = 'euclidean'
    elif distance_type == "L1 distance":
        print("Using L1 distance")
        ord_arg = 1
        cdist_arg = 'cityblock'
    else:
        raise Exception("Invalid distance type request")


    iteration_num = 0
    while (np.linalg.norm(c - c_old, ord=ord_arg) > 1e-6 and iteration_num <= 200) or iteration_num == 0: 
        # record previous c;
        c_old = copy.deepcopy(c) 
        # end semi demo code

        dist = sp.distance.cdist(pixels, np.array(c), cdist_arg)  # prints out the distance from each pixel to the cluster center. Pick the min.

        # cluster assignment
        # get the index of the minimum value to find out which cluster center it belongs to
        cluster_assignment_arr = np.array(np.argmin(dist, axis=1))  # look across rows

        # Demo code
        # Recompute centroids;
        m = pixels.shape[0]
        P = csc_matrix((np.ones(m), (np.arange(0, m, 1), cluster_assignment_arr)), shape=(m, k))
        count = P.sum(axis=0)

        c = np.array(((P.T).dot(pixels)).T / count).T
        # End of Demo code
        
        # Handle instances where high k values cause empty clusters. Simply remove them.
        # Also, check before/after if a modification was made and adjust the c_old accordingly to prevent a dimension mismatch error.
        c_preClean = c.copy()
        #print(c)
        c = c[~np.isnan(c).any(axis=1), :]
        c_postClean = c.copy()
        k = len(c)  # reduce the cluster # to represent the nonzero clusters remaining

        if not np.array_equal(c_postClean,c_preClean):
            c_old = c + 10
            print("not equal")

        error = np.linalg.norm(c - c_old, ord=ord_arg)

        print(f"Iteration {iteration_num} | Running error: {error}")

        iteration_num += 1

    centroid = c

    #print(P)

    #print("Final Centroids")
    #print(centroid)

    return cluster_assignment_arr, centroid, iteration_num



if __name__ == '__main__':
    replication_count = 1

    os_dir = os.path.dirname(__file__)
    print(os_dir)
    image_path_arr = [os.path.join(os_dir, r"data/football.bmp"), os.path.join(os_dir, r"data/uni.bmp"), os.path.join(os_dir, r"data/bird.bmp")]
    distance_type_arr = ["L2 distance", "L1 distance"]
    #distance_type_arr = ["L1 distance"]

    for image_path in image_path_arr:
        image = Image.open(image_path)
        image_pixels =  np.array(image.getdata())  # each row corresponds to 1 pixel, each row has 3 cols

        
        for distance_type in distance_type_arr:
            for q in range(0, replication_count):
                k_arr = [2,4,8,16]
                #k_arr = [4]
                for k in k_arr:
                    print(f"On k {k}, distance: {distance_type}, {image_path}")
                    time1 = time.time()
                    class_val, centroid, iteration_num = kmeans(image_pixels, k, distance_type)
                    time2 = time.time()
                    print(f"Time to converge: {time2-time1} seconds | Iterations: {iteration_num - 1}")

                    centroid_list_rounded = list(np.rint(centroid))
                    centroid_list_rounded_1 = []
                    for i in centroid_list_rounded:
                        list_i = list(i)
                        list_i_2 = [int(j) for j in list_i]
                        centroid_list_rounded_1.append(list_i_2)

                    class_val = list(class_val)
                    # Take the assignments and reconstruct the image for testing purposes
                    new_image_arr = []
                    for i in range(0, len(class_val)):
                        new_image_arr.append(tuple(centroid_list_rounded_1[class_val[i] - 1]))

                        if i % 100000 == 0:
                            print(f"On: {i}")

                    # Write image
                    new_image = Image.new("RGB", image.size)  # make a new template image first
                    new_image.putdata(new_image_arr)  # put in data
                    new_image.save(image_path.replace("data/","output/").replace(".bmp",f"_{distance_type}_{k}_{q}.bmp"))  # save it