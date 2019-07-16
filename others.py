import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

# parameter
n = 200
dim = 5
class_num = 3
x = 3 * (np.random.rand(n, 4) - 0.5)
x = np.hstack([x, np.ones((n, 1))])
W = np.array([[2, -1, 0.5], [-3, 2, 1], [1, 2, 3]])
x_with_error = np.dot(np.hstack([x[:, 0:2].reshape((n, 2)), np.ones((n, 1))]), W.T) + 0.5 * np.random.randn(n, 3)
maxlogit, y = x_with_error.max(axis=1), x_with_error.argmax(axis=1)
lam = 2
alpha = 0.8
w = np.ones((dim, 3))
# --- steepest gradient method ---
arr_sgm = []
while 1:
    grad = 2 * lam * w
    obj_fun = lam * np.trace(np.dot(w.T, w))
    L = 0
    for i in range(n):
        xi = x[i].reshape((dim, 1))
        yi = y[i].reshape((1, 1))
        exps = np.exp(np.dot(w.T, xi))
        softmax = exps / np.sum(exps)
        for j in range(class_num):
            if (j == yi):
                grad[:, yi] += ((softmax[yi] - 1).reshape((1, 1)) * xi).reshape((5, 1, 1))
            else:
                grad[:, j] += (softmax[j].reshape((1, 1)) * xi).reshape((5,))
        L -= (np.log(softmax[yi]))
    grad /= n
    obj_fun += L
    obj_fun /= n
    w -= grad
    arr_sgm.append(np.asscalar(obj_fun))
    if np.linalg.norm(grad) < 0.001:
        break
print("steepest gradient method")
print(w)


# --- newton method ---
w = np.ones((dim, 3))
step = 0
arr_newton = []
while 1:
    grad = 2 * lam * w
    obj_fun = lam * np.trace(np.dot(w.T, w))
    L = 0
    hessian = [2 * lam * np.eye(5), 2 * lam * np.eye(5), 2 * lam * np.eye(5)]
    for i in range(n):
        xi = x[i].reshape((dim, 1))
        yi = y[i].reshape((1, 1))
        exps = np.exp(np.dot(w.T, xi))
        softmax = exps / np.sum(exps)
        for j in range(class_num):
            if (j == yi):
                grad[:, yi] += ((softmax[yi] - 1).reshape((1, 1)) * xi).reshape((5, 1, 1))
            else:
                grad[:, j] += (softmax[j].reshape((1, 1)) * xi).reshape((5,))
            hessian[j] += (1 - softmax[j]) * softmax[j] * np.dot(xi, xi.T)
        L -= (np.log(softmax[yi]))
    grad /= n
    obj_fun += L
    obj_fun /= n
    for j in range(class_num):
        hessian[j] /= n
    for j in range(class_num):
        d = np.dot(np.linalg.inv(hessian[j]), grad[:, j])
        w[:, j] -= alpha * d
    #print(hessian)
    arr_newton.append(np.asscalar(obj_fun))
    if np.linalg.norm(grad) < 0.001:
        print(grad)
        break


print("--- compare ---")
tmp1 = arr_sgm[len(arr_sgm)-1]
arr_sgm -= tmp1*np.ones((len(arr_sgm)))
plt.plot(np.arange(0, len(arr_sgm), 1), arr_sgm)
tmp2 = arr_newton[len(arr_newton)-1]
arr_newton -= tmp2*np.ones((len(arr_newton)))
plt.plot(np.arange(0, len(arr_newton), 1), arr_newton)
#print(arr_sgm)
#plt.plot(np.arange(0, len(arr_newton), 1), arr_newton)
plt.show()