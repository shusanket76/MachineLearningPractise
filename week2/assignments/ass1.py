import numpy as np
from autils import load_data
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from public_tests import test_c1


x,y=load_data()

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


prediction = model.predict(x[607].reshape(1,400))
yhat=0
print(prediction)
if prediction>=0.5:
    print("hi")
    yhat = 1
else:
    yhat=0
print(y[607],yhat)