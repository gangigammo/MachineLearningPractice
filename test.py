import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

# a = sym.Symbol('a')
# b = sym.Symbol('b')
# x = sym.Symbol('x')
# y = sym.Symbol('y')
# w = sym.Symbol('w')
# wt = sym.Symbol('wt')
# lam = sym.Symbol('lam')
# X = sym.Symbol('X')
# C = sym.Symbol('C')
# lam = sym.Symbol('lam')
# w = sym.Symbol('w')
#
# ideltaj = (-X*sym.exp(C))/(1 + sym.exp(C))
# print(ideltaj)

# w -= lam * deltaj
#     if np.sum(np.abs(w)) < 0.01:
#         break

#shape(x = n * 4,y= n * 1,w = 4 * 1)
#parameter
n = 200
x = 3 * (np.random.rand(n, 4) - 0.5)
x = np.hstack([x, np.ones((n, 1))])
dim = 5
y = (2 * x[:, 0].reshape((n, 1)) - 1 * x[:, 1].reshape((n, 1)) + 0.5 + 0.5 * np.random.randn(n, 1)) > 0
y = 2 * y - 1
lam = 2
alpha = 0.01

#--- steepest gradient method ---
print("--- steepest gradient method ---")
w = np.ones((dim, 1))
step = 0
arr_sgm = []
while 1:
    grad = 2 * lam * w
    obj_fun = lam * np.dot(w.T, w)
    for i in range(n):
        xi = x[i].reshape(dim, 1)
        yi = y[i].reshape(1, 1)
        Xin = -y[i]*x[i].reshape(dim, 1)
        exp = np.exp(-yi * np.dot(w.T, xi))
        grad += -exp/(1 + exp)*xi*yi
        obj_fun += np.log(1 + exp)
    w -= grad * alpha
    arr_sgm.append(np.asscalar(obj_fun))
    step += 1
    if np.linalg.norm(grad) < 0.001:
        #print(grad)
        break
print("step:",end="")
print(step)
print(w)

# --- newton method ---
print("--- newton method ---")
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
        #print(grad)
        break
print("step:",end="")
print(step)
print(w)

print(arr_sgm)
print(arr_newton)

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

x = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[2, -1, 0.5], [-3,  2,   1], [1,  2, 3]])

x_with_error = np.dot(np.hstack([x[:, 0:2].reshape((n, 2)), np.ones((n, 1))]), W.T) + 0.5 * np.random.randn(n, 3)
maxlogit, y = x_with_error.max(axis=1), x_with_error.argmax(axis=1)


#for k in range(1):
# print(np.shape(y))
# print(np.shape(w))
# print(np.shape(x))
# print(np.shape(y[0]))
# print(np.shape(w.T))
# print(np.shape(x[0].T))

# for k in range(100):
#     j = 0
#     for i in range(4):
#         j += np.log(1 + np.exp(-y[i].reshape((1, 1)) * w.T * x[i].T.reshape((4, 1))))
#     j += lam * w.T * w
#     print(sum)

# while 1:
#     deltaj = 0
#     for i in range(4):
#         deltaj += sym.diff(np.log(1 + np.exp(-y[i].reshape((1, 1)) * w.T * x[i].T.reshape((4, 1)))),w)
#     deltaj += sym.diff(lam * w.T * w,w)
#     deltaj *= -1
#     w -= lam * deltaj
#     if np.sum(np.abs(w)) < 0.01:
#         break
# print(w)