from sklearn.datasets import make_blobs
from multiclasslab import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

classes=4
m=100
centers = [[-5,2], [-2,-2], [1,2], [5,-2]]
std = 1.0
xtrain,ytrain = make_blobs(n_samples=m, centers=centers)
# print(xtrain)
# plt_mc(xtrain,ytrain, classes, centers, std=1.0)
print(ytrain[:10])
print(xtrain.shape)
print(ytrain.shape)

model = Sequential(
    [
        Dense(2, activation="relu", name="layer1"),
        Dense(4, activation="linear", name="linearlayer2")
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.legacy.Adam(0.01))

model.fit(xtrain,ytrain,epochs=200)
w1,b1= model.get_layer("layer1").get_weights()
print(w1)
print("====================")
print(b1)


plt_cat_mc(xtrain,ytrain, model, classes)
plt_layer_relu(xtrain,ytrain,w1,b1,classes)

l2 = model.get_layer("linearlayer2")
w2,b2 = l2.get_weights()
x12 = np.maximum(0,np.dot(xtrain,w1)+b1)
