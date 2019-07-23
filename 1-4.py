import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

#parameter
n = 200
dim = 5
x = 3 * (np.random.rand(n, 4) - 0.5)
x = np.hstack([x, np.ones((n, 1))])
W = np.array([[2, -1, 0.5], [-3,  2,   1], [1,  2,   3]])
x_with_error = np.dot(np.hstack([x[:, 0:2].reshape((n, 2)), np.ones((n, 1))]), W.T) + 0.5 * np.random.randn(n, 3)
maxlogit, y = x_with_error.max(axis=1), x_with_error.argmax(axis=1)
print(np.shape(x_with_error))
print(np.shape(y))
lam = 2
alpha = 0.01

# --- steepest method ---
w = np.ones((dim, 1))
step = 0
arr_newton = []
while 1:
    grad = 2 * lam * w
    #hessian = 2 * lam * np.eye(5)
    obj_fun = lam * np.dot(w.T, w)
    for i in range(n):
        xi = x[i].reshape(dim, 1)
        yi = y[i].reshape(1, 1)
        exp = np.exp(-yi * np.dot(w.T, xi))
        pi = 1/(1+exp)
        grad += -exp/(1 + exp)*xi*yi
        #hessian += pi*(1-pi)*np.dot(xi, xi.T) * yi**2
        obj_fun += np.log(1 + exp)
    #d = -np.dot(np.linalg.inv(hessian), grad)
    d = -grad
    w = w + d
    arr_newton.append(np.asscalar(obj_fun))
    step += 1
    if np.linalg.norm(d) < 0.001:
        print(grad)
        break
print("--- newton method ---")
print("step:")
print(step)
print(w)


# --- newton method ---
w = np.ones((dim, 1))
step = 0
arr_newton = []
while 1:
    grad = 2 * lam * w
    hessian = 2 * lam * np.eye(5)
    obj_fun = lam * np.dot(w.T, w)
    for i in range(n):
        xi = x[i].reshape(dim, 1)
        yi = y[i].reshape(1, 1)
        exp = np.exp(-yi * np.dot(w.T, xi))
        pi = 1/(1+exp)
        grad += -exp/(1 + exp)*xi*yi
        hessian += pi*(1-pi)*np.dot(xi, xi.T) * yi**2
        obj_fun += np.log(1 + exp)
    d = -np.dot(np.linalg.inv(hessian), grad)
    w = w + d
    arr_newton.append(np.asscalar(obj_fun))
    step += 1
    if np.linalg.norm(d) < 0.001:
        print(grad)
        break
print("--- newton method ---")
print("step:")
print(step)
print(w)
