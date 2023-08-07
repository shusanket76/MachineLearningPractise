import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.keras.activations import sigmoid



# =====================LINEAR STARTS=======================================
# xtrain = np.array([[1.0],[2.0]])
# ytrain = np.array([[300.0],[500.0]])

# linear_layer = Dense(units=1, activation="linear")
# print(linear_layer.get_weights())
# print()
# w,b = linear_layer.get_weights()
# print(w,b)
# set_w  = np.array([[200]])
# set_b = np.array([100])
# linear_layer(xtrain[0].reshape(1,1))
# linear_layer.set_weights([set_w,set_b])
# print(linear_layer.get_weights())
# a1 = linear_layer(xtrain[0].reshape(1,1))
# print(a1)
# alin = np.dot(set_w,xtrain[0].reshape(1,1))+set_b
# print("hello break")
# print(alin)
# print("second break")

# predictiontf = linear_layer(xtrain)
# predictionnp = np.dot(xtrain,set_w)+set_b
# print(predictionnp)
# print("third break")
# print(predictiontf)
# LINEAR ENDS=========================================================


