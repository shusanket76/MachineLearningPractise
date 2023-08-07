import numpy as np
import copy
import math

xtrain = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
ytrain = np.array([460, 232, 178])

# DISPLAYING THE SHPAE OF THE NUMPY ARRAY
# print(xtrain.shape)

# initializing the parameters value
binit = 785.1811367994083
winit = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# IMPLEMENTED THE LEARNING ALGORITHM
def predict(x,w,b):
    p = np.dot(x,w)+b
    return p

# x_vec = xtrain[0:]
# print(x_vec)
pred = predict(xtrain,winit,binit)
print(pred)


def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost = 0.0
    for i in range(m):
        yhat = np.dot(x[i],w)+b
        cost=cost+(yhat-y[i])**2
    cost = cost/(2*m)
    return cost

cost = compute_cost(xtrain,ytrain,winit,binit)
print(cost)


def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw = np.zeros((n,))
    dj_db=0

    for i in range(m):
        err = (np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+err*x[i,j]
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_db, dj_dw


def gradient_descent(x,y,winit,binit,cost,gradient,alpha,num_iteration):
    J_history=[]
    w = copy.deepcopy(winit)
    b = binit
    for i in range(num_iteration):
        dj_db, dj_dw=gradient(x,y,w,b)
        w = w-alpha*dj_dw
        b=b-alpha*dj_db
        print(i)
        print("hi")
        if i<1000000:
            J_history.append(cost(x,y,w,b))
        if i% math.ceil(num_iteration / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w,b,J_history
    


initial_w = np.zeros_like(winit)
initial_b=0
iterations = 10000
alpha = 5.0e-7

wfinal,bfinal,Jhist = gradient_descent(xtrain, ytrain, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
# print(wfinal,bfinal)
# print(Jhist)
m,_=xtrain.shape
