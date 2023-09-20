import os
import numpy as np
import pandas as pd
from os.path import abspath, exists
import scipy
from sklearn.cluster import KMeans

# Demo code function (modified a little):
def import_edges():
    os_dir = os.path.dirname(__file__)
    f_path = os.path.join(os_dir, r"data/edges.txt")
    lines = []
    if exists(f_path):
        with open(f_path) as graph_file:
            
            for line in graph_file:
                temp_line_arr = line.split()
                # Based on Piazza post by TA, remove edges that connect to themselves
                if temp_line_arr[0] != temp_line_arr[1]:
                    lines.append(temp_line_arr)
    return np.unique(np.array(lines).astype(int), axis=0)  # Based on Piazza post by TA, remove edges that are duplicates


# Demo code function (modified a bit):
def read_nodes(unique_edges):
    unique_edges_list = list(unique_edges)
    #print(unique_edges_list)
    os_dir = os.path.dirname(__file__)
    f_path = os.path.join(os_dir, r"data/nodes.txt")
    line_arr = []
    if exists(f_path):
        with open(f_path) as fid:
            for line in fid:
                temp_line_arr = line.split("\t")
                if int(temp_line_arr[0]) in unique_edges_list:  # only include nodes that are present in the edges file as per TA
                    line_arr.append(temp_line_arr)
    return np.array(line_arr)


if __name__ == "__main__":
    edges = import_edges()  # [[i,j],[i,j]]
    #print(f"edges shape: {edges.shape}")

    edges_i = edges[:,0]  # demo
    edges_j = edges[:,1]  # demo

    unique_i = np.unique(edges_i)
    unique_j = np.unique(edges_j)
    unique_edges = np.unique(np.concatenate((unique_i, unique_j), axis=None))  # flatten the arrays, join them, and find new uniques

    nodes = read_nodes(unique_edges)
    nodes_list = nodes[:,0]

    # construct A matrix
    edges_i_list = list(edges_i)
    edges_j_list = list(edges_j)

    edges_df = pd.DataFrame()
    edges_df["i"] = edges_i
    edges_df["j"] = edges_j
    A_list = []
    # iterate over i (rows of matrix) with default value to be 0. Change to 1 if a value is found.
    for nodes_i in nodes_list:
        nodes_int = int(nodes_i)

        # get nodes with i->j connections
        edges_temp_df = edges_df[edges_df["i"] == nodes_int]
        edges_j_vals = list(edges_temp_df["j"])

        # get nodes with j->i connections
        edges_temp_df = edges_df[edges_df["j"] == nodes_int]
        edges_i_vals = list(edges_temp_df["i"])

        edges_vals = edges_j_vals + edges_i_vals
        temp_dict = dict.fromkeys(nodes_list, 0)  # default value of 0

        for vals in edges_vals:
            temp_dict[str(vals)] += 1

        temp_dict_arr = np.fromiter(temp_dict.values(), dtype=int)
        
        A_list.append(temp_dict_arr)
    A = np.array(A_list)

    A = (A + np.transpose(A))/2  # demo
    A = np.asmatrix(A)  # matrix form is important for L in np.linalg.eig

    D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1) # demo

    L = D - A
    L = np.array(L) # demo covert to array
    
    # Demo code (modified):
    # eigendecompoosition
    v, x_original= np.linalg.eig(L)
    v = np.real(v)  # TA Piazza discussion suggestion
    x_original = np.real(x_original)
    idx_sorted = np.argsort(v)


    k_arr = [2,5,10,20]  # for part 1
    #k_arr = list(np.arange(2, 51))  # part 2 code:
    TOTAL_mismatch_k = []
    TOTAL_mismatch_val = []
    for k in k_arr:
        print(f"On k: {k}")
        x = x_original
        x = x[:, idx_sorted][:,:k] # modified demo, select the k smallest eigenvectors

        # k-means
        kmeans = KMeans(n_clusters=k).fit(x)  # demo code
        c_idx = kmeans.labels_  # demo code

        # Compute mismatch
        mismatch_df = pd.DataFrame()
        mismatch_df["nodes"] = nodes_list
        mismatch_df["true_labels"] = list(nodes[:,2])
        mismatch_df["cluster_labels"] = c_idx

        all_clusters = list(mismatch_df["cluster_labels"].unique())

        total_size = 0
        majority_label_count_total = 0
        for cluster in all_clusters:
            mismatch_df_copy = mismatch_df.copy()
            mismatch_df_copy = mismatch_df_copy[mismatch_df_copy["cluster_labels"] == cluster]

            # Now, mismatch df will have the same cluster_labels, yet different true_labels
            majority_label, majority_label_count = scipy.stats.mode(np.array(mismatch_df_copy["true_labels"]))
            size = mismatch_df_copy["cluster_labels"].shape[0]

            cluster_mismatch = 1 - (majority_label_count / size)

            print(f"For cluster {cluster}, mismatch rate of {cluster_mismatch}")

            total_size += size
            majority_label_count_total += majority_label_count

        TOTAL_mismatch = 1 - (majority_label_count_total / total_size)
        print(f"TOTAL {k}, mismatch rate of {TOTAL_mismatch}")

        TOTAL_mismatch_k.append(k)
        TOTAL_mismatch_val.append(float(TOTAL_mismatch))
    
    TOTAL_mismatch_df = pd.DataFrame()
    TOTAL_mismatch_df["Cluster Count"] = TOTAL_mismatch_k
    TOTAL_mismatch_df["Mismatch Rate"] = TOTAL_mismatch_val

    #TOTAL_mismatch_df.sort_values(by=["Mismatch Rate"], inplace=True)  # part 2 code:
    print(TOTAL_mismatch_df)