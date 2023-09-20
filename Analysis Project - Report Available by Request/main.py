import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sklearn.metrics
import time

import faiss  # KNN testing - python 3.5 only?
from scipy.stats import mode

import matplotlib.pyplot as plt 

os_dir = os.path.dirname(__file__)
# os_dir_data = os.path.join(os_dir, r"data/")
os_dir_data = os.path.join(os_dir, r"data/original/")

test_csv_path = os_dir_data + "1000_test.csv"
train_csv_path = os_dir_data + "1000_train.csv"

header_list = ["# label","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26"]

######## Read in the original Data and downsample (not used for TAs/Grading/Final)

n_samples = 10000

test_df = pd.read_csv(test_csv_path, header=0, names=header_list)
print(test_df.shape)
test_df = test_df.sample(n=n_samples)
test_df.to_csv(test_csv_path.replace("/original",""), index=False)

train_df = pd.read_csv(train_csv_path, names=header_list)
print(train_df.shape)
train_df = train_df.sample(n=n_samples*2)
train_df.to_csv(train_csv_path.replace("/original",""), index=False)

# print(train_df)

######## End Downsample code

test_df = pd.read_csv(test_csv_path, names=header_list, skiprows=[0])  # skip row 0 because we are creating our own header
train_df = pd.read_csv(train_csv_path, names=header_list, skiprows=[0])  # skip row 0 because we are creating our own header

# if "f26" in test_df.columns:
#     test_df.drop(columns=["f26"], inplace=True)
#     print("Dropped f26.")
# if "f26" in train_df.columns:
#     train_df.drop(columns=["f26"], inplace=True)
#     print("Dropped f26.")

# print(test_df)

# test_df_quantile = test_df.quantile(q=[0,.01,.1,.9,.99,1])
# train_df_quantile = train_df.quantile(q=[0,.01,.1,.9,.99,1])

# print(test_df_quantile)
# print(train_df_quantile)

####### scale data

df_columns = list(test_df.columns)

df_columns_features = df_columns
df_columns_label = df_columns[0]
df_columns_features.pop(0)


print(df_columns_features)
print(type(df_columns_features))

scaler = StandardScaler()
test_df[df_columns_features] = scaler.fit_transform(test_df[df_columns_features])
train_df[df_columns_features] = scaler.fit_transform(train_df[df_columns_features])

####### end scale data

####### Separate the label from the normal dataframe

test_label_arr = np.array(test_df[df_columns_label])
test_df = test_df.drop(columns=df_columns_label)

train_label_arr = np.array(train_df[df_columns_label])
train_df = train_df.drop(columns=df_columns_label)
#######

############## Logistic Regression

print("Starting Logistic Regression")
# Cross validation to find the best regularization strength first
clf_LR = LogisticRegressionCV(cv=5,Cs=20, penalty="l2", random_state=1)
clf_LR.fit(train_df, train_label_arr)

LR_best_regularization = float(clf_LR.C_)

print(f"Best LR Regularization term: {LR_best_regularization}")

# Fitting model
print("Fitting Model")
clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
clf_LR.fit(train_df, train_label_arr.ravel())
pred_labels = np.array(clf_LR.predict(test_df))

# test_df_accuracy = pd.DataFrame(columns=["Actual Label","Predicted Label", "Accuracy"])
# test_df_accuracy["Actual Label"] = test_label_arr
# test_df_accuracy["Predicted Label"] = pred_labels
# test_df_accuracy["Accuracy"] = test_df_accuracy["Actual Label"] - test_df_accuracy["Predicted Label"]

report_LR = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels, digits=4)
print(report_LR)

###################

# # predict_proba attempt to optimize sample_weights for edge prediction cases
# pred_probs_LR = clf_LR.predict_proba(train_df)  # check on train data to reweight
# print(pred_probs_LR.shape)

# pred_probs_LR_diff = np.abs(np.diff(pred_probs_LR, axis=1))

# # High values need to lose weight. Low values need to gain weight.
# min_weight = .4
# max_weight = .6

# pred_probs_LR_diff_proportion = (1 - pred_probs_LR_diff) * (max_weight-min_weight) + min_weight
# pred_probs_LR_diff_proportion = pred_probs_LR_diff_proportion.ravel()

# print(pred_probs_LR_diff_proportion.shape)

# print("new range ",np.min(pred_probs_LR_diff_proportion), np.max(pred_probs_LR_diff_proportion))

# clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
# clf_LR.fit(train_df, train_label_arr.ravel(), sample_weight=pred_probs_LR_diff_proportion)
# pred_labels_adj = np.array(clf_LR.predict(test_df))

# report_LR_adj = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels_adj, digits=4)
# print(report_LR_adj)

# ############## 

# ###### test
# # predict_proba attempt to optimize sample_weights for edge prediction cases
# pred_probs_LR = clf_LR.predict_proba(train_df)  # check on train data to reweight
# print(pred_probs_LR.shape)

# pred_probs_LR_diff = np.abs(np.diff(pred_probs_LR, axis=1))

# # High values need to lose weight. Low values need to gain weight.
# min_weight = .3
# max_weight = .7

# pred_probs_LR_diff_proportion = (1 - pred_probs_LR_diff) * (max_weight-min_weight) + min_weight
# pred_probs_LR_diff_proportion = pred_probs_LR_diff_proportion.ravel()

# print(pred_probs_LR_diff_proportion.shape)

# print("new range ",np.min(pred_probs_LR_diff_proportion), np.max(pred_probs_LR_diff_proportion))

# clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
# clf_LR.fit(train_df, train_label_arr.ravel(), sample_weight=pred_probs_LR_diff_proportion)
# pred_labels_adj = np.array(clf_LR.predict(test_df))

# report_LR_adj = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels_adj, digits=4)
# print(report_LR_adj)

# ######

# ###### test
# # predict_proba attempt to optimize sample_weights for edge prediction cases
# pred_probs_LR = clf_LR.predict_proba(train_df)  # check on train data to reweight
# print(pred_probs_LR.shape)

# pred_probs_LR_diff = np.abs(np.diff(pred_probs_LR, axis=1))

# # High values need to lose weight. Low values need to gain weight.
# min_weight = .2
# max_weight = .8

# pred_probs_LR_diff_proportion = (1 - pred_probs_LR_diff) * (max_weight-min_weight) + min_weight
# pred_probs_LR_diff_proportion = pred_probs_LR_diff_proportion.ravel()

# print(pred_probs_LR_diff_proportion.shape)

# print("new range ",np.min(pred_probs_LR_diff_proportion), np.max(pred_probs_LR_diff_proportion))

# clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
# clf_LR.fit(train_df, train_label_arr.ravel(), sample_weight=pred_probs_LR_diff_proportion)
# pred_labels_adj = np.array(clf_LR.predict(test_df))

# report_LR_adj = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels_adj, digits=4)
# print(report_LR_adj)

# ######

# ###### test
# # predict_proba attempt to optimize sample_weights for edge prediction cases
# pred_probs_LR = clf_LR.predict_proba(train_df)  # check on train data to reweight
# print(pred_probs_LR.shape)

# pred_probs_LR_diff = np.abs(np.diff(pred_probs_LR, axis=1))

# # High values need to lose weight. Low values need to gain weight.
# min_weight = .1
# max_weight = .9

# pred_probs_LR_diff_proportion = (1 - pred_probs_LR_diff) * (max_weight-min_weight) + min_weight
# pred_probs_LR_diff_proportion = pred_probs_LR_diff_proportion.ravel()

# print(pred_probs_LR_diff_proportion.shape)

# print("new range ",np.min(pred_probs_LR_diff_proportion), np.max(pred_probs_LR_diff_proportion))

# clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
# clf_LR.fit(train_df, train_label_arr.ravel(), sample_weight=pred_probs_LR_diff_proportion)
# pred_labels_adj = np.array(clf_LR.predict(test_df))

# report_LR_adj = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels_adj, digits=4)
# print(report_LR_adj)

# ######

###### test
# predict_proba attempt to optimize sample_weights for edge prediction cases
pred_probs_LR = clf_LR.predict_proba(train_df)  # check on train data to reweight
print(pred_probs_LR.shape)

pred_probs_LR_diff = np.abs(np.diff(pred_probs_LR, axis=1))

# High values need to lose weight. Low values need to gain weight.
min_weight = 0
max_weight = 1

pred_probs_LR_diff_proportion = (1 - pred_probs_LR_diff) * (max_weight-min_weight) + min_weight
pred_probs_LR_diff_proportion = pred_probs_LR_diff_proportion.ravel()

print(pred_probs_LR_diff_proportion.shape)

print("new range ",np.min(pred_probs_LR_diff_proportion), np.max(pred_probs_LR_diff_proportion))

clf_LR = LogisticRegression(C=LR_best_regularization,penalty="l2", random_state=1)
clf_LR.fit(train_df, train_label_arr.ravel(), sample_weight=pred_probs_LR_diff_proportion)
pred_labels_adj = np.array(clf_LR.predict(test_df))

report_LR_adj = sklearn.metrics.classification_report(y_true = test_label_arr, y_pred = pred_labels_adj, digits=4)
print(report_LR_adj)

######



class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype('float32'), k=self.k)
        votes = self.y[indices]
        # predictions = np.array([np.argmax(np.bincount(x)).astype(float) for x in votes])
        predictions = np.apply_along_axis(lambda x: mode(x)[0], 1, votes)
        return predictions

# ############## Sklearn K-NN
# # choose k between 1 to 31
# k_range = range(20, 24)
# k_scores = []# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation

# start_time_knn = time.time()
# print("Starting KNN:")
# for k in k_range:
#     print(f"On: {k}")
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, train_df, train_label_arr, cv=5, scoring='accuracy')
#     k_scores.append(scores.mean())# plot to see clearly


# end_time_knn = time.time()
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()

# best_k = np.argmax(k_scores) + 1

# print(f"Best k: {best_k}")



# print(f"KNN Completion time: {end_time_knn - start_time_knn} seconds")
# ############## End Sklearn K-NN

############## Start Faiss KNN

train_df_nparr_ccont = np.ascontiguousarray(np.array(train_df)).astype('float32')
test_df_nparr_ccont = np.ascontiguousarray(np.array(test_df)).astype('float32')
train_label_arr_ccont = np.ascontiguousarray(train_label_arr).astype('float32')


#https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb

start_time_faiss = time.time()
k_range = range(23, 24)  # range(1, 31)
faiss_k_scores = []
# accuracy_score_arr = []
print("Starting faiss KNN:")
for k in k_range:
    print(f"On: {k}")
    faiss_knn = FaissKNeighbors(k=k)
    # faiss_knn.fit(train_df_nparr_ccont, train_label_arr_ccont)   # uncomment if needed to ditch k-fold experiment
    skf = StratifiedKFold()
    skf.get_n_splits(train_df, train_label_arr)  # Create test indicies and train indicies from the TRAINING data, so that we can validate.

    fold = 1
    accuracy_score_fold_arr = []
    for train_index, validate_index in skf.split(train_df, train_label_arr):
        X_train, X_validate = train_df.iloc[train_index], train_df.iloc[validate_index]  # pull from the train DF to make new train and validate indicies
        y_train, y_validate = np.take(train_label_arr,train_index), np.take(train_label_arr,validate_index)   # pull from the train DF to make new train and validate indicies
        
        X_train_cont = np.ascontiguousarray(X_train).astype('float32')
        y_train_cont = np.ascontiguousarray(y_train).astype('float32')
        X_validate_cont = np.ascontiguousarray(X_validate).astype('float32')

        # Get predictions to compare
        faiss_knn.fit(X_train_cont, y_train_cont) #Training the model
        y_pred = faiss_knn.predict(X_validate_cont)

        accuracy_score_fold_arr.append(sklearn.metrics.accuracy_score(y_validate, y_pred))
        fold += 1

    accuracy_val = np.mean(accuracy_score_fold_arr)
    print(f"Accuracy Value: {accuracy_val}")
    faiss_k_scores.append(accuracy_val)


# print(faiss_pred.shape)

end_time_faiss = time.time()

plt.plot(k_range, faiss_k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

best_k = np.argmax(faiss_k_scores) + 1

print(f"Best k: {best_k}")

print(f"Faiss KNN Completion time: {end_time_faiss - start_time_faiss} seconds")

############## End Faiss KNN

