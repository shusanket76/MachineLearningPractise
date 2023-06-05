# ALL ABOUT NUMPY
import numpy as np
import time


# creates array starting from 2 to 10
a = np.arange(2,10)
print(a)
print(a.shape)
# creates random values
b = np.random.random_sample(20)
print(b)


# slicing
print(b[2:5])
# sum of array
print(np.sum(b))
# mean
print(np.mean(b))
# ========================================7
np1 = np.array([1,2])
np2 = np.array([3,4])
print(np1+np2)

# dotproduct = The dot product multiplies the values in two vectors element-wise and then sums the result. Vector dot product requires the dimensions of the two vectors to be the same.
print(np.dot(np1,np2))
# ==========================================
# Matrices
mat = np.zeros((1,5))
print(mat)

mat1 = np.arange(12).reshape(-1,3)
print(mat1)