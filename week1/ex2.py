import copy, math
import numpy as np
import matplotlib.pyplot as plt
xtrain = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
ytrain = np.array([460,232,178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict_single_loop(x,w,b):
    n=x.shape[0]
    p=0
    for i in range(n):
        p_i=x[i]*w[i]
        p=p+p_i
    p=p+b
    return p

x_vec = xtrain[0]
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f_wb)
# print(x_vec)

def compute_cost(X,y,w,b):
    m=X.shape[0]
    print(X.shape)
    print(m)
    cost =0.0
    for i in range(m):
        f_wb = np.dot(X[i],w)+b
        cost = cost+(f_wb-y[i])**2
    cost = cost/(2*m)
    return cost

cost = compute_cost(xtrain,ytrain,w_init,b_init)
print(cost)

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        err = (np.dot(X[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+err*X[i,j]
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
gd = compute_gradient(xtrain,ytrain,w_init,b_init)
print(gd)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha,num_iters):
    J_history=[]
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db  
                # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history #return final w,b and J history for graphing
    
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(xtrain, ytrain, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)

m=xtrain.shape[0]
for i in range(m):
    print(f"prediction: {np.dot(xtrain[i], w_final) + b_final:0.2f}, target value: {ytrain[i]}")