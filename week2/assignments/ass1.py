import numpy as np
from autils import load_data
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from public_tests import test_c1, test_c2
from utils import sigmoid

x,y=load_data()

# ============================TF MODEL=======================================
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(25, activation="sigmoid"),
        Dense(15, activation="sigmoid"),
        Dense(1, activation="sigmoid"),
    ], name='shusanket_model'
)


model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(x,y,epochs=20)

[layer1,layer2,layer3] = model.layers
w1,b1 = layer1.get_weights()
w2,b2 = layer2.get_weights()
w3,b3 = layer3.get_weights()
# prediction = model.predict(x[607].reshape(1,400))
# yhat=0
# print(prediction)
# if prediction>=0.5:
#     print("hi")
#     yhat = 1
# else:
#     yhat=0
# print(y[607],yhat)
# ===========================NUMPY MODEL===============================================

# def my_dense(in_data, weight, bias, activationFunction):
#     units = weight.shape[1]
#     a_out = np.zeros(units)
#     for i in range(units):
#         w = weight[:,i]
#         z = np.dot(w,in_data)+bias[i]
#         a_out[i]= activationFunction(z)

#     return a_out

# x_tst = 0.1*np.arange(1,3,1).reshape(2,)  # (1 examples, 2 features)
# W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
# b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
# A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)


# def my_seq(x,w1,b1,w2,b2,w3,b3):
#     a1 = my_dense(x,w1,b1,sigmoid)
#     a2 = my_dense(a1,w2,b2,sigmoid)
#     a3 = my_dense(a2,w3,b3,sigmoid)
#     return a3

# w1 = np.zeros([400,25])
# b1 = np.zeros([400])
# w2 =np.zeros([25,15])
# b2 = np.zeros(25)
# w3 = np.zeros([15,1])
# b3 = np.zeros([15])
# pred = my_seq(x[710],w1,b1,w2,b2,w3,b3)
# yhat = 0
# if pred>=0.5:
#     yhat = 1
# else:
#     yhat=0

# print(pred)
# print(yhat, y[710])
# ============================MATRIX MULTIPLICATION==============================================/?
def mydense(x,w,b,g):
    z= np.matmul(x,w)+b
    return g(z)
def my_seq(x,w1,b1,w2,b2,w3,b3):
    a1 = mydense(x,w1,b1,sigmoid)
    a2 = mydense(a1,w2,b2,sigmoid)
    a3 = mydense(a2,w3,b3,sigmoid)
    return a3
pred = my_seq(x,w1,b1,w2,b2,w3,b3)
print(pred)