# -*- coding: utf-8 -*-
import gzip
import numpy as np
import os
import requests 
from sklearn.svm import SVC, LinearSVC
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from scipy import interp
from itertools import cycle
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def load_mnist(train_data=True, test_data=False):
    RESOURCES = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']
    if (os.path.isdir('data') == 0):
        os.mkdir('data')
    if (os.path.isdir('data/mnist') == 0):
        os.mkdir('data/mnist')
    for name in RESOURCES:
        if (os.path.isfile('data/mnist/'+name) == 0):
            url = 'http://yann.lecun.com/exdb/mnist/'+name
            r = requests.get(url, allow_redirects=True)
            open('data/mnist/'+name, 'wb').write(r.content)
    return get_images(train_data, test_data), get_labels(train_data, test_data)
    #train_data = get_images(train_data)
    #test_data = get_images(test_data)
    #train_lable = get_labels(train_data)
    #test_lable = get_labels(test_data)
    #return train_data, test_data, train_lable, test_lable

load_mnist(train_data=True, test_data=True)

def get_images(train_data=True, test_data=False):
    to_return = []
    if train_data:
        with gzip.open('data/mnist/train-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            train_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(train_images > 127, 1, 0))

    if test_data:
        with gzip.open('data/mnist/t10k-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            test_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(test_images > 127, 1, 0))
    arr_return = np.array(to_return[0])
    return arr_return


def get_labels(train_data=True, test_data=False):
    to_return = []
    if train_data:
        with gzip.open('data/mnist/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            train_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(train_labels)
    if test_data:
        with gzip.open('data/mnist/t10k-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            test_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(test_labels)
    arr_return = np.array(to_return[0])
    return arr_return

train_images = get_images(train_data=True, test_data=False)
test_images = get_images(train_data=False, test_data=True)
train_lables = get_labels(train_data=True, test_data=False)
test_lables = get_labels(train_data=False, test_data=True)

from matplotlib import pyplot
for i in range(9):  
  pyplot.subplot(330 + 1 + i)
  pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))
  pyplot.show()

def imgtodf(img):
  arr = np.empty((0 ,784), int)
  for i in img:
    array_1d = np.array(i).flatten()
    array_1d = array_1d.reshape(1,784)
    arr = np.append(arr, np.array(array_1d), axis=0)
    df = pd.DataFrame(arr)
    print
  return df

X_train = imgtodf(train_images)
X_test = imgtodf(test_images)

X_train.to_csv('/content/drive/MyDrive/pattern//Train.csv', index=False)
X_test.to_csv('/content/drive/MyDrive/pattern//Test.csv', index=False)

xtrain = pd.read_csv('/content/drive/MyDrive/pattern//Train.csv')
xtest = pd.read_csv('/content/drive/MyDrive/pattern//Test.csv')

ytrain = pd.DataFrame(train_lables)
ytest =  pd.DataFrame(test_lables)

"""# **All Features**

## **All Features SVM RBF**
"""

# Commented out IPython magic to ensure Python compatibility.
# 
# 
# %%time
# 
# classifier = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))
# classifier.fit(xtrain,ytrain)
# 
# 
# ypred_rbf = classifier.predict(xtest)
# ypred_rbf_dfunction = classifier.fit(xtrain,ytrain).decision_function(xtest)
# print("Accuracy:", metrics.accuracy_score(ytest, ypred_rbf))
# 
#

#confusion matrix
print('confusion ' , confusion_matrix(ytest, ypred_rbf))

#precision
print('Precision:' ,  precision_score(ytest, ypred_rbf, average='weighted'))


#recall

print('Recal :' , recall_score(ytest, ypred_rbf, average='weighted'))


#f1 score
print('f1 score  ' , f1_score(ytest,ypred_rbf, average= 'weighted'))

ybinerized_labels = label_binarize(ytest, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_rbf_dfunction[:, i])
    roc_auc[i] =  (fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_rbf_dfunction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""## **All Features Linear SVC**"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# clftotal = OneVsRestClassifier(make_pipeline(StandardScaler(),
#                      LinearSVC(random_state=0, tol=1e-5)))
# clftotal.fit(xtrain, ytrain)

ypred_linear = clftotal.predict(xtest)
ypredlinearrbf_dfunction = clftotal.fit(xtrain,ytrain).decision_function(xtest)
print("Accuracy:", metrics.accuracy_score(ytest, ypred_linear))

#confusion matrix
print('confusion ' , confusion_matrix(ytest, ypred_linear))

#precision
print('Precision:' ,  precision_score(ytest, ypred_linear, average='weighted'))


#recall

print('Recal :' , recall_score(ytest, ypred_linear, average='weighted'))


#f1 score
print('f1 score  ' , f1_score(ytest,ypred_linear, average= 'weighted'))

# Binarize the output
ybinerized_labels = label_binarize(ytest, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypredlinearrbf_dfunction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypredlinearrbf_dfunction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""## **All Features Decision Tree**"""

clf = OneVsRestClassifier(DecisionTreeClassifier())
clf.fit(xtrain,ytrain)
ypred_tree = clf.predict(xtest)
ypred_dec = clf.fit(xtrain, ytrain).predict_proba(xtest)
print("Accuracy:", metrics.accuracy_score(ytest, ypred_tree))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_tree))

print('precision :' ,precision_score(ytest, ypred_tree , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_tree,average= 'weighted'))
 

print('f1score :', f1_score(ytest,ypred_tree, average= 'weighted'))

ypred_dec_df = pd.DataFrame(ypred_dec)
ypred_dec_df= ypred_dec_df.fillna(0)
ypred_dec = ypred_dec_df.to_numpy()

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_dec[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_dec.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""# **PCA**"""

pca_40 = PCA(n_components=40)

xtrain_pca40 = pca_40.fit(xtrain)

xtest_pca40 = pca_40.fit(xtest)

pca_100 = PCA(n_components=100)

xtrain_pca100 = pca_100.fit(xtrain)

xtest_pca100 = pca_100.fit(xtest)

xtest_pca100 = xtest_pca100.transform(xtest)
xtrain_pca40 = xtrain_pca40.transform(xtrain)

xtest_pca40 = xtest_pca40.transform(xtest)
xtrain_pca100 = xtrain_pca100.transform(xtrain)

"""## **PCA 40**

### **PCA 40 - RBF SVC**
"""

# Commented out IPython magic to ensure Python compatibility.
# 
# 
# %%time
# 
# classifier = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))
# classifier.fit(xtrain_pca40,ytrain)
# 
# 
# ypred_rbf40 = classifier.predict(xtest_pca40)
# ypred_rbf_dfunction40 = classifier.fit(xtrain_pca40,ytrain).decision_function(xtest_pca40)
# print("Accuracy:", metrics.accuracy_score(ytest, ypred_rbf40))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_rbf40))

print('precision :' ,precision_score(ytest, ypred_rbf40 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_rbf40,average= 'weighted'))

print('f1score :', f1_score(ytest,ypred_rbf40, average= 'weighted'))

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_rbf_dfunction40[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_rbf_dfunction40.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""### **PCA 40 - Decision Tree**"""

clf40 = OneVsRestClassifier(DecisionTreeClassifier())
clf40.fit(xtrain_pca40,ytrain)
ypred_tree_40 = clf40.predict(xtest_pca40)
ypred_dec40 = clf40.fit(xtrain_pca40, ytrain).predict_proba(xtest_pca40)
print("Accuracy:", metrics.accuracy_score(ytest, ypred_tree_40))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_tree_40))

print('precision :' ,precision_score(ytest, ypred_tree_40 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_tree_40,average= 'weighted'))
 

print('f1score :', f1_score(ytest,ypred_tree_40, average= 'weighted'))

ypred_dec_df40 = pd.DataFrame(ypred_dec40)
ypred_dec_df40= ypred_dec_df40.fillna(0)
ypred_dec40 = ypred_dec_df40.to_numpy()

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_dec40[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_dec40.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""### **PCA 40 - Linear SVC**"""

# Commented out IPython magic to ensure Python compatibility.
# 
# %%time
# clflinearpca40 = OneVsRestClassifier(make_pipeline(StandardScaler(),
#                      LinearSVC(random_state=0, tol=1e-5)))
# clflinearpca40.fit(xtrain_pca40, ytrain)
# ypred_linear_pca40 = clflinearpca40.predict(xtest_pca40)
# ypredlinearrbf_dfunction_pca40 = clflinearpca40.fit(xtrain_pca40,ytrain).decision_function(xtest_pca40)
# print("Accuracy:", metrics.accuracy_score(ytest, ypred_linear_pca40))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_linear_pca40))

print('precision :' ,precision_score(ytest, ypred_linear_pca40 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_linear_pca40,average= 'weighted'))

print('f1score :', f1_score(ytest,ypred_linear_pca40, average= 'weighted'))

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypredlinearrbf_dfunction_pca40[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypredlinearrbf_dfunction_pca40.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""## **PCA 100**

### **PCA 100 - RBF SVC**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# classifier100 = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))
# classifier100.fit(xtrain_pca100,ytrain)
# 
# 
# ypred_rbf100 = classifier100.predict(xtest_pca100)
# ypred_rbf_dfunction100 = classifier100.fit(xtrain_pca100,ytrain).decision_function(xtest_pca100)
# print("Accuracy:", metrics.accuracy_score(ytest, ypred_rbf100))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_rbf100))

print('precision :' ,precision_score(ytest, ypred_rbf100 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_rbf100,average= 'weighted'))

print('f1score :', f1_score(ytest,ypred_rbf100, average= 'weighted'))

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_rbf_dfunction100[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_rbf_dfunction100.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""### **PCA 100 - Decision Tree**"""

clf100 = OneVsRestClassifier(DecisionTreeClassifier())
clf100.fit(xtrain_pca100,ytrain)
ypred_tree_100 = clf100.predict(xtest_pca100)
ypred_dec100 = clf100.fit(xtrain_pca100, ytrain).predict_proba(xtest_pca100)
print("Accuracy:", metrics.accuracy_score(ytest, ypred_tree_100))

ypred_dec_df100 = pd.DataFrame(ypred_dec100)
ypred_dec_df100= ypred_dec_df100.fillna(0)
ypred_dec100 = ypred_dec_df100.to_numpy()

print('confusion matrix :' ,confusion_matrix(ytest, ypred_tree_100))

print('precision :' ,precision_score(ytest, ypred_tree_100 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_tree_100,average= 'weighted'))

print('f1score :', f1_score(ytest,ypred_tree_100, average= 'weighted'))

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypred_dec100[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypred_dec100.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""### **PCA 100 - Linear SVC**"""

clflinearpca100 = OneVsRestClassifier(make_pipeline(StandardScaler(),
                     LinearSVC(random_state=0, tol=1e-5)))
clflinearpca100.fit(xtrain_pca100, ytrain)

ypred_linear_pca100 = clflinearpca100.predict(xtest_pca100)
ypredlinearrbf_dfunction_pca100 = clflinearpca100.fit(xtrain_pca100,ytrain).decision_function(xtest_pca100)
print("Accuracy:", metrics.accuracy_score(ytest, ypred_linear_pca100))

print('confusion matrix :' ,confusion_matrix(ytest, ypred_linear_pca100))

print('precision :' ,precision_score(ytest, ypred_linear_pca100 , average= 'weighted'))

print('recall :' , recall_score(ytest, ypred_linear_pca100,average= 'weighted'))

print('f1score :', f1_score(ytest,ypred_linear_pca100, average= 'weighted'))

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ybinerized_labels[:, i], ypredlinearrbf_dfunction_pca100[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(' {0}  is :'.format(i),roc_auc[i], "\n")
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ybinerized_labels.ravel(), ypredlinearrbf_dfunction_pca100.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
n_classes = 10
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

"""# **BOUNUS**

## **Linear SVC , K = 5**
"""

xtrain_pca100 = pd.DataFrame(xtrain_pca100)
xtest_pca100 = pd.DataFrame(xtest_pca100)
Xconcat_100 = [xtrain_pca100, xtest_pca100]
yconcat = [ytrain, ytest]
x_100 = pd.concat(Xconcat_100)
y = pd.concat(yconcat)

k = 5
kf = KFold(n_splits=k, random_state=None)
model_SVM_linear = OneVsRestClassifier(make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)))

accuracyscore_linear = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = x_100.iloc[train_index,:],x_100.iloc[test_index,:]
  ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
  
  model_SVM_linear.fit(xtrain, ytrain)
  predvalues_linear = model_SVM_linear.predict(xtest)
  acc_linear = metrics.accuracy_score(predvalues_linear, ytest)
  accuracyscore_linear.append(acc_linear)

avg_accuracyscore_linear = sum(accuracyscore_linear)/k

print('accuracy of each fold - {}'.format(accuracyscore_linear))
print('Avg accuracy : {}'.format(avg_accuracyscore_linear))

"""## **Linear SVC , K = 7**"""

k = 7
kf = KFold(n_splits=k, random_state=None)
model_SVM_linear = OneVsRestClassifier(make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)))

# Commented out IPython magic to ensure Python compatibility.
# %%time 
# accuracyscore_linear = []
# for train_index, test_index in kf.split(X):
#   xtrain , xtest =  x_100.iloc[train_index,:], x_100.iloc[test_index,:]
#   ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
#   
#   model_SVM_linear.fit(xtrain, ytrain)
#   predvalues_linear = model_SVM_linear.predict(xtest)
#   acc_linear = metrics.accuracy_score(predvalues_linear, ytest)
#   accuracyscore_linear.append(acc_linear)
# 
# avg_accuracyscore_linear = sum(accuracyscore_linear)/k
# 
# print('accuracy of each fold - {}'.format(accuracyscore_linear))
# print('Avg accuracy : {}'.format(avg_accuracyscore_linear))

"""## **Linear SVC , K = 10**"""

k = 10
kf = KFold(n_splits=k, random_state=None)
model_SVM_linear = OneVsRestClassifier(make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)))

# Commented out IPython magic to ensure Python compatibility.
# %%time 
# accuracyscore_linear = []
# for train_index, test_index in kf.split(X):
#   xtrain , xtest =  x_100.iloc[train_index,:], x_100.iloc[test_index,:]
#   ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
#   
#   model_SVM_linear.fit(xtrain, ytrain)
#   predvalues_linear = model_SVM_linear.predict(xtest)
#   acc_linear = metrics.accuracy_score(predvalues_linear, ytest)
#   accuracyscore_linear.append(acc_linear)
# 
# avg_accuracyscore_linear = sum(accuracyscore_linear)/k
# 
# print('accuracy of each fold - {}'.format(accuracyscore_linear))
# print('Avg accuracy : {}'.format(avg_accuracyscore_linear))

"""## **RBF SVC , K= 5**"""

xtrain_pca100 = pd.DataFrame(xtrain_pca100)
xtest_pca100 = pd.DataFrame(xtest_pca100)
Xconcat_100 = [xtrain_pca100, xtest_pca100]
yconcat = [ytrain, ytest]
x_100 = pd.concat(Xconcat_100)
y = pd.concat(yconcat)

k = 5
kf = KFold(n_splits=k, random_state=None)
model_rbf = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))

accuracyscore_rbf = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = x_100.iloc[train_index,:],x_100.iloc[test_index,:]
  ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
  
  model_rbf.fit(xtrain, ytrain)
  predvalues_rbf = model_rbf.predict(xtest)
  acc_rbf = metrics.accuracy_score(predvalues_rbf, ytest)
  accuracyscore_rbf.append(acc_rbf)

avg_accuracyscore_rbf = sum(accuracyscore_rbf)/k

print('accuracy of each fold - {}'.format(accuracyscore_rbf))
print('Avg accuracy : {}'.format(avg_accuracyscore_rbf))

"""## **RBF SVC , K= 7**"""

k = 7
kf = KFold(n_splits=k, random_state=None)
model_rbf = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))

accuracyscore_rbf = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = x_100.iloc[train_index,:],x_100.iloc[test_index,:]
  ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
  
  model_rbf.fit(xtrain, ytrain)
  predvalues_rbf = model_rbf.predict(xtest)
  acc_rbf = metrics.accuracy_score(predvalues_rbf, ytest)
  accuracyscore_rbf.append(acc_rbf)

avg_accuracyscore_rbf = sum(accuracyscore_rbf)/k

print('accuracy of each fold - {}'.format(accuracyscore_rbf))
print('Avg accuracy : {}'.format(avg_accuracyscore_rbf))

"""## **RBF SVC , K= 10**"""

k = 10
kf = KFold(n_splits=k, random_state=None)
model_rbf = OneVsRestClassifier(SVC(kernel='rbf', random_state=1))

accuracyscore_rbf = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = x_100.iloc[train_index,:],x_100.iloc[test_index,:]
  ytrain , ytest = y.iloc[train_index] , y.iloc[test_index]
  
  model_rbf.fit(xtrain, ytrain)
  predvalues_rbf = model_rbf.predict(xtest)
  acc_rbf = metrics.accuracy_score(predvalues_rbf, ytest)
  accuracyscore_rbf.append(acc_rbf)

avg_accuracyscore_rbf = sum(accuracyscore_rbf)/k

print('accuracy of each fold - {}'.format(accuracyscore_rbf))
print('Avg accuracy : {}'.format(avg_accuracyscore_rbf))

"""## **DecisionTree , k = 5**"""

xconcat = [xtrain, xtest]
yconcat = [ytrain, ytest]
Xtree = pd.concat(xconcat)
ytree = pd.concat(yconcat)

k = 5
kf = KFold(n_splits=k, random_state=None)
model_tree = OneVsRestClassifier(DecisionTreeClassifier())

accuracyscore_tree = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = Xtree.iloc[train_index,:],Xtree.iloc[test_index,:]
  ytrain , ytest = ytree.iloc[train_index] , ytree.iloc[test_index]
  
  model_tree.fit(xtrain, ytrain)
  predvalues_tree = model_tree.predict(xtest)
  acc_tree = metrics.accuracy_score(predvalues_tree, ytest)
  accuracyscore_tree.append(acc_tree)

avg_accuracyscore_tree = sum(accuracyscore_tree)/k

print('accuracy of each fold - {}'.format(accuracyscore_tree))
print('Avg accuracy : {}'.format(avg_accuracyscore_tree))

"""## **DecisionTree , k = 7**"""

k = 7
kf = KFold(n_splits=k, random_state=None)
model_tree = OneVsRestClassifier(DecisionTreeClassifier())

accuracyscore_tree = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = Xtree.iloc[train_index,:],Xtree.iloc[test_index,:]
  ytrain , ytest = ytree.iloc[train_index] , ytree.iloc[test_index]
  
  model_tree.fit(xtrain, ytrain)
  predvalues_tree = model_tree.predict(xtest)
  acc_tree = metrics.accuracy_score(predvalues_tree, ytest)
  accuracyscore_tree.append(acc_tree)

avg_accuracyscore_tree = sum(accuracyscore_tree)/k

print('accuracy of each fold - {}'.format(accuracyscore_tree))
print('Avg accuracy : {}'.format(avg_accuracyscore_tree))

"""## **DecisionTree , k = 10**"""

k = 10
kf = KFold(n_splits=k, random_state=None)
model_tree = OneVsRestClassifier(DecisionTreeClassifier())

accuracyscore_tree = []
for train_index, test_index in kf.split(X):
  xtrain , xtest = Xtree.iloc[train_index,:],Xtree.iloc[test_index,:]
  ytrain , ytest = ytree.iloc[train_index] , ytree.iloc[test_index]
  
  model_tree.fit(xtrain, ytrain)
  predvalues_tree = model_tree.predict(xtest)
  acc_tree = metrics.accuracy_score(predvalues_tree, ytest)
  accuracyscore_tree.append(acc_tree)

avg_accuracyscore_tree = sum(accuracyscore_tree)/k

print('accuracy of each fold - {}'.format(accuracyscore_tree))
print('Avg accuracy : {}'.format(avg_accuracyscore_tree))