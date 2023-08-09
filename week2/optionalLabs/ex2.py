import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

from lab_coffee_utils import load_coffee_data, plt_roast
x,y = load_coffee_data()
norm1 = tf.keras.layers.Normalization(axis=-1)

norm1.adapt(x)
xn = norm1(x)
xt = np.tile(xn,(1000,1))
yt = np.tile(y,(1000,1))
tf.random.set_seed(1234)
model = Sequential(
[
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
model.summary()
w1,b1 = model.get_layer("layer1").get_weights()
model.compile(
    loss =tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01))
model.fit(xt,yt,epochs=10)
wnew,bnew = model.get_layer("layer1").get_weights()
print(w1,b1)
print("------------------------------")
print(wnew,bnew)

