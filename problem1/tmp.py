import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

# parameter
n = 200
dim = 5
x = 3 * (np.random.rand(n, 4) - 0.5)
x = np.hstack([x, np.ones((n, 1))])
W = np.array([[2, -1, 0.5], [-3, 2, 1], [1, 2, 3]])
x_with_error = np.dot(np.hstack([x[:, 0:2].reshape((n, 2)), np.ones((n, 1))]), W.T) + 0.5 * np.random.randn(n, 3)
maxlogit, y = x_with_error.max(axis=1), x_with_error.argmax(axis=1)
print(np.shape(x))
print(np.shape(x_with_error))
print(np.shape(y))

lam = 2
alpha = 0.01
w = np.ones((dim, 3))
step = 0
while 1:
    grad = 2 * lam * (w[:, 0] + w[:, 1] + w[:, 2]).reshape((5, 1))
    obj_fun = lam * np.trace(np.dot(w.T, w))
    L = 0
    for i in range(n):
        yi = y[i].reshape((1, 1))
        xi = x[i].reshape((dim, 1))
        exps = np.exp(np.dot(w.T, xi))
        softmax = exps / np.sum(exps)
        grad += (softmax[yi] - 1).reshape((1, 1)) * xi
        L -= (np.log(softmax[yi]))
    grad /= n
    obj_fun += L
    obj_fun /= n
    step += 1
# while 1:
#     grad = 2 * lam * (w[:, 0] + w[:, 1] + w[:, 2]).reshape((5, 1))
#     obj_fun = lam * np.trace(np.dot(w.T, w))
#     L = 0
#     for i in range(n):
#         yi = y[i].reshape((1, 1))
#         xi = x[i].reshape((dim, 1))
#         exps = np.exp(np.dot(w.T, xi))
#         softmax = exps / np.sum(exps)
#         grad += (softmax[yi] - 1).reshape((1, 1)) * xi
#         L -= (np.log(softmax[yi]))
#     grad /= n
#     obj_fun += L
#     obj_fun /= n
#     step += 1