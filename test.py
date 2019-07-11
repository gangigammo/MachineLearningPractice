import numpy as np
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
y = (2 * x[:, 0].reshape((n,1)) - 1 * x[:, 1].reshape((n,1)) + 0.5 + 0.5 * np.random.randn(n, 1)) > 0
y = 2 * y - 1
win = np.ones((dim,1))
lamin = 0.01
I5 = np.eye(5)
step = 0
print(win)
# --- steepest gradient method ---
# while 1:
#     deltaj = 2 * lamin * win
#     for i in range(n):
#         Cin = 0
#         Xin = -y[i]*x[i].reshape(4,1)
#         #print(Xin)
#         for j in range(4):
#             Cin -= win[j] * x[i][j] * y[i].reshape(1, 1)
#         deltaj += (math.exp(Cin))/(1 + math.exp(Cin))*Xin
#     win -= deltaj *lamin
#     #print(np.linalg.norm(deltaj))
#     if np.linalg.norm(deltaj) < 0.01:
#         print(deltaj)
#         break
#     #print(win)
# print(win)

# --- newton method ---
while 1:
    grad = 2 * lamin * win
    hessian = 2 * lamin * np.eye(5)
    for i in range(n):
        xi = x[i].reshape(dim,1)
        yi = y[i].reshape(1,1)
        exp = np.exp(-yi * np.dot(win.T, xi))
        pi = 1/(1+exp)
        grad += -exp/(1 + exp)*xi*yi
        hessian += pi*(1-pi)*np.dot(xi, xi.T) * yi**2
    d = -np.dot(np.linalg.inv(hessian), grad)
    win = win + d
    step += 1
    if np.linalg.norm(d) < 0.01:
        print(grad)
        break
print(step)
print(win)





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