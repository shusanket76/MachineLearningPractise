import numpy as np


def load_data():
    x = np.load("data/X.npy")
    y = np.load("data/Y.npy")
    print(len(x))
    x = x[0:1000]
    y=y[0:1000]
    return x,y
load_data()

