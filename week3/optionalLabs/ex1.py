from sklearn.datasets import make_blobs
from multiclasslab import *
import numpy as np

classes=4
m=100
centers = [[-5,2], [-2,-2], [1,2], [5,-2]]
std = 1.0
xtrain,ytrain = make_blobs(n_samples=m, centers=centers)
# print(xtrain)
# plt_mc(xtrain,ytrain, classes, centers, std=1.0)
print(np.unique(ytrain))