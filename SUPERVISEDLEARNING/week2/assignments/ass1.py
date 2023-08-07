import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy 
import math


xtrain,ytrain = load_data()
print(xtrain.shape)
print(ytrain.shape)


def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for singleX in range(len(x)):
        f = (w*x[singleX])+b
        cost += (f-y[singleX])**2
    totalCost = (cost)/(2*m)
    return totalCost

# initial_w = 2
# intial_b=1
# cost = compute_cost(xtrain,ytrain,initial_w,intial_b)
# print(cost)


def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_db=0
    dj_dw=0
    for single in range(m):
        f = (w*x[single])+b
        dj_db += (f-y[single])
        dj_dw += (f-y[single])*x[single]
    dj_db /=m
    dj_dw /=m
    return dj_dw,dj_db

# initial_w=0.2
# initial_b=0.2
# tempdj_dw,tempdj_db = compute_gradient(xtrain,ytrain,initial_w,initial_b)
# print(tempdj_db,tempdj_dw)

def gradient_descent(x,y,w_in, b_in,cost_function,gradient_function, num_iterations,alpha):
    m = len(x)
    cost_history = []
    w_histor = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iterations):
        dj_dw,dj_db =compute_gradient(x,y,w,b)
        w = w - (alpha*dj_dw)
        b = b - (alpha*dj_db)

        if i<100000:
            cost = cost_function(x,y,w,b)
            cost_history.append(cost)
    return w, b, cost_history 

w_in = 0
b_in = 0
iterations = 2000
alpha = 0.01
w,b,costhistory = gradient_descent(xtrain,ytrain,w_in,b_in,compute_cost,compute_gradient,iterations,alpha)
print(costhistory[0], costhistory[len(costhistory)-1])
print(w,b)
