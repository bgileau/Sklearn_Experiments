import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import os

from skimage import io
from scipy.io import loadmat

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, metrics, decomposition, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import random

import seaborn
from matplotlib.colors import ListedColormap

# Demo code function (modified) with piazza suggestions too
def run_classifier(clf, Xtrain, ytrain, Xtest, ytest):
    # ## training error
    ypred_train = clf.predict(Xtrain)
    matched_train = ypred_train == ytrain
    acc_train = sum(matched_train)/len(matched_train)

    ypred_test = clf.predict(Xtest)
    matched_test = ypred_test == ytest
    acc_test = sum(matched_test)/len(matched_test)
 
    confusion_matrix_train = metrics.confusion_matrix(ytrain,ypred_train)  # piazza
    #print(confusion_matrix_train)

    confusion_matrix_test = metrics.confusion_matrix(ytest,ypred_test)  # piazza
    #print(confusion_matrix_test)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('the training accuracy: '+ str(round(acc_train, 4)))
    print('confusion matrix for training:')
    print('          predicted 1       predicted 2')
    print(f"true 1        {confusion_matrix_train[0][0]}                {confusion_matrix_train[0][1]}")
    print(f"true 2        {confusion_matrix_train[1][0]}                {confusion_matrix_train[1][1]}")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('the testing accuracy: '+ str(round(acc_test, 4)))
    print('confusion matrix for testing:')
    print('          predicted 1       predicted 2')
    print(f"true 1        {confusion_matrix_test[0][0]}                {confusion_matrix_test[0][1]}")
    print(f"true 2        {confusion_matrix_test[1][0]}                {confusion_matrix_test[1][1]}")

# Create paths to subjects
os_dir = os.path.dirname(__file__)
os_dir_data = os.path.join(os_dir, r"data/")

# a and b, part 1
marriage_df = pd.read_csv(os_dir_data + "marriage.csv", header=None)

marriage_label = marriage_df.iloc[:,-1].ravel()
marriage_predictors = marriage_df.iloc[:,:-1]

################## 1.a:
marriage_predictors = StandardScaler().fit_transform(marriage_predictors)  # demo
Xtrain_, Xtest_, ytrain_, ytest_ = model_selection.train_test_split(marriage_predictors,marriage_label,train_size=.80, test_size=.20)  # demo code
classifier_list = ["LR", "KNN", "NB"]
for classifier in classifier_list:
    
    print("=====================================")
    if classifier == "LR":
        print("Logistic Regression Results for (a)")
        clf = LogisticRegression(max_iter=200, solver='liblinear').fit(Xtrain_, ytrain_)
    elif classifier == "KNN":
        print("KNN Results for (a)")
        clf = neighbors.KNeighborsClassifier(2).fit(Xtrain_, ytrain_)
    elif classifier == "NB":
        print("NB Results for (a)")
        clf = GaussianNB().fit(Xtrain_, ytrain_)

    print("Running classifier")
    run_classifier(clf, Xtrain_, ytrain_, Xtest_, ytest_)


################## 1.b:
print("Part B")
print("PCA")
pca = decomposition.PCA(n_components=2).fit(Xtrain_)

print(pca)

pca_trans_x_train = pca.transform(Xtrain_)
pca_trans_x_test = pca.transform(Xtest_)

#print(pca_trans_x_train, pca_trans_x_test)
print(pca_trans_x_train.shape, Xtrain_.shape)

cm = plt.cm.RdBu  # demo
cm_bright = ListedColormap(['#FF0000', '#0000FF']) # demo

classifier_list = ["LR", "KNN", "NB"]
for classifier in classifier_list:
    print("=====================================")
    if classifier == "LR":
        print("Logistic Regression Results for (b)")
        clf = LogisticRegression(max_iter=200, solver='liblinear').fit(pca_trans_x_train, ytrain_)
        levels=0
    elif classifier == "KNN":
        print("KNN Results for (b)")
        clf = neighbors.KNeighborsClassifier(2).fit(pca_trans_x_train, ytrain_)
        levels=1
    elif classifier == "NB":
        print("NB Results for (b)")
        clf = GaussianNB().fit(pca_trans_x_train, ytrain_)
        levels=1
    
    print("Train Data Plot")
    ypred_train = clf.predict(pca_trans_x_train)
    h = .02  # step size in the mesh
    
    ###### Demo code start:
    x_min, x_max = pca_trans_x_train[:, 0].min() - .5, pca_trans_x_train[:, 0].max() + .5
    y_min, y_max = pca_trans_x_train[:, 1].min() - .5, pca_trans_x_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8, levels=levels, cmap=cm)
    plt.scatter(pca_trans_x_train[:,0], pca_trans_x_train[:,1], c=ypred_train, cmap=cm_bright,edgecolors='k')

    plt.show()
    
    print("Test Data Plot")
    ypred_test= clf.predict(pca_trans_x_test)
    
    x_min, x_max = pca_trans_x_test[:, 0].min() - .5, pca_trans_x_test[:, 0].max() + .5
    y_min, y_max = pca_trans_x_test[:, 1].min() - .5, pca_trans_x_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8, levels=levels, cmap=cm)
    plt.scatter(pca_trans_x_test[:,0], pca_trans_x_test[:,1], c=ypred_test, cmap=cm_bright,edgecolors='k')

    plt.show()
    ###### Demo code end:

######## part 2
### a

def run_classifier_part2(clf, Xtrain, Ytrain, Xtest, Ytest):
    print("Predicting training data")
    ypred_train = clf.predict(Xtrain)
    report_train = metrics.classification_report(y_true = Ytrain, y_pred = ypred_train)
    print("Train Data report")
    print(report_train)
    confusion_matrix_train = metrics.confusion_matrix(Ytrain,ypred_train)  # piazza
    print(confusion_matrix_train)

    print("Predicting Test Data")
    ypred_test = clf.predict(Xtest)
    report_test = metrics.classification_report(y_true = Ytest, y_pred = ypred_test)
    print("Test Data report")
    print(report_test)

    
    confusion_matrix_test = metrics.confusion_matrix(Ytest,ypred_test)  # piazza
    print(confusion_matrix_test)

minst_data = loadmat(os_dir_data + "mnist_10digits.mat")

Xtrain_ = minst_data["xtrain"] / 255
Ytrain_ = (minst_data["ytrain"]).ravel()
Xtest_ = minst_data["xtest"] / 255
Ytest_ = (minst_data["ytest"]).ravel()

# Generate 20,000 random numbers, get the unique ones, and then select the first 5000 (to avoid double counting/sampling)
indices_train = np.random.randint(0, Ytrain_.shape,size=20000)
indices_test = np.random.randint(0, Ytest_.shape,size=20000)

indices_train = np.array(np.unique(indices_train))[:5000]
indices_test = np.array(np.unique(indices_test))[:5000]

Xtrain_5000 = Xtrain_[indices_train]
Ytrain_5000 = Ytrain_[indices_train]
Xtest_5000 = Xtest_[indices_test]
Ytest_5000 = Ytest_[indices_test]

print(Xtrain_5000.shape, Ytrain_5000.shape, Xtest_5000.shape, Ytest_5000.shape)

classifier_list = ["KNN", "LR", "SVM", "kSVM", "NN"]
for classifier in classifier_list:

    Xtrain_use = Xtrain_
    Ytrain_use = Ytrain_
    Xtest_use = Xtest_
    Ytest_use = Ytest_

    
    print("=====================================")
    if classifier == "LR":
        print("Logistic Regression Results for (a)")
        clf = LogisticRegression(max_iter=200, solver='liblinear').fit(Xtrain_use, Ytrain_use)
    elif classifier == "KNN":
        print("KNN Results for (a)")
        Xtrain_use = Xtrain_5000
        Ytrain_use = Ytrain_5000
        Xtest_use = Xtest_5000
        Ytest_use = Ytest_5000
        clf = neighbors.KNeighborsClassifier(4).fit(Xtrain_use, Ytrain_use)
    elif classifier == "SVM":
        print("SVM Results for (a)")
        Xtrain_use = Xtrain_5000
        Ytrain_use = Ytrain_5000
        Xtest_use = Xtest_5000
        Ytest_use = Ytest_5000
        clf = SVC(kernel="linear", C=0.025).fit(Xtrain_use, Ytrain_use)
    elif classifier == "kSVM":
        print("kSVM Results for (a)")
        Xtrain_use = Xtrain_5000
        Ytrain_use = Ytrain_5000
        Xtest_use = Xtest_5000
        Ytest_use = Ytest_5000
        clf = SVC(gamma="scale", C=1).fit(Xtrain_use, Ytrain_use)
    elif classifier == "NN":
        print("NN Results for (a)")
        clf = MLPClassifier(alpha=1, max_iter=1000).fit(Xtrain_use, Ytrain_use)

    
    print("Running classifier")
    run_classifier_part2(clf, Xtrain_use, Ytrain_use, Xtest_use, Ytest_use)
