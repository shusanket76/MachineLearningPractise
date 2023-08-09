import numpy as np
import tensorflow as tf
from lab_coffee_utils import load_coffee_data
from lab_utils_common import sigmoid

x,y=load_coffee_data()
print(x.shape,y.shape)

norm1 = tf.keras.layers.Normalization(axis=-1)
norm1.adapt(x)
xn = norm1(x)


def hiddenlayers(entry, weight, bias, activationFunction):
    units = weight.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w=weight[:,j]
        z = np.dot(w,entry)+bias[j]
        a_out[j] = activationFunction(z)
    return a_out

def buildNetwork(x,w1,b1,w2,b2):
    a1 = hiddenlayers(x,w1,b1,sigmoid)
    a2 = hiddenlayers(a1,w2,b2,sigmoid)
    return a2
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def mypredict(x,w1,b1,w2,b2):
    m = x.shape[0]
    p=np.zeros((m,1))
    for i in range(m):
        p[i,0] = buildNetwork(x[i],w1,b1,w2,b2)
    print(p)
    return p
mypredict(xn,W1_tmp,b1_tmp,W2_tmp,b2_tmp)