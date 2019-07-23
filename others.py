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
alpha = 0.02
w = np.ones((dim, 3))
# --- steepest gradient method ---
arr_sgm = []
step = 0
while 1:
        #w_history.append(w)
        grad = np.zeros_like(w)
        obj_fun = lam * np.trace(np.dot(w.T, w))
        L = 0
        for i in range(n):
            yi = y[i].reshape((1,1))
            xi = x[i].reshape((dim,1))
            c = np.max(np.dot(w.T, xi)) # avoid overflow
            exps = np.exp(np.dot(w.T, xi) - c)
            softmax = exps / np.sum(exps)
            #print(softmax)
            for j in range(class_num):
                if j == yi:
                    grad[:,yi] += ((softmax[yi] - 1).reshape((1,1)) * xi).reshape((5,1,1))
                else:
                    grad[:,j] += (softmax[j].reshape((1,1)) * xi).reshape((5,))
            L -= (np.log(softmax[yi]))
            #print(grad)
        grad /= n
        grad += 2 * lam * w
        L /= n
        obj_fun += L
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(obj_fun)))
        #obj_fun_history.append(np.asscalar(obj_fun))
        if (np.linalg.norm(grad) < 1e-6):
            break
        w -= alpha * grad
        step += 1

print("steepest gradient method")
print("step:" + str(step))
print(w)


# --- newton method ---
w = np.ones((dim, 3))
step = 0
arr_newton = []
# while 1:
#     #grad = 2 * lam * w
#     grad = np.zeros_like(w)
#     obj_fun = lam * np.trace(np.dot(w.T, w))
#     L = 0
#     hessian = [np.zeros((5,5)), np.zeros((5,5)), np.zeros((5,5))]
#     for i in range(n):
#         xi = x[i].reshape((dim, 1))
#         yi = y[i].reshape((1, 1))
#         maximum = np.max(np.dot(w.T, xi))
#         exps = np.exp(np.dot(w.T, xi) - maximum)
#         softmax = exps / np.sum(exps)
#         for j in range(class_num):
#             if j == yi:
#                 grad[:, yi] += ((softmax[yi] - 1).reshape((1, 1)) * xi).reshape((5, 1, 1))
#             else:
#                 grad[:, j] += (softmax[j].reshape((1, 1)) * xi).reshape((5,))
#             hessian[j] += (1 - softmax[j]) * softmax[j] * np.dot(xi, xi.T)
#         L -= (np.log(softmax[yi]))
#     grad /= n
#     grad = grad + 2 * lam * w
#     L /= n
#     obj_fun += L
#     for j in range(class_num):
#         hessian[j] /= n
#         hessian[j] += 2 * lam * np.eye(5)
#     for j in range(class_num):
#         d = np.dot(np.linalg.inv(hessian[j]), grad[:, j])
#         w[:, j] -= alpha * d
#     #print(hessian)
#     arr_newton.append(np.asscalar(obj_fun))
#     if np.linalg.norm(grad) < 1e-6:
#         print(grad)
#         break
#     step += 1
while 1:
        #w_history.append(w)
        hessian = [np.zeros((5,5)), np.zeros((5,5)), np.zeros((5,5))]
        grad = np.zeros_like(w)
        obj_fun = lam * np.trace(np.dot(w.T, w))
        L = 0
        # batch
        for i in range(n):
            yi = y[i].reshape((1,1))
            xi = x[i].reshape((5,1))
            c = np.max(np.dot(w.T, xi)) # avoid overflow
            exps = np.exp(np.dot(w.T, xi) - c)
            softmax = exps / np.sum(exps)
            for j in range(class_num):
                if (j ==yi):
                    grad[:,yi] += ((softmax[yi] - 1).reshape((1,1)) * xi).reshape((5,1,1))
                else:
                    grad[:,j] += (softmax[j].reshape((1,1)) * xi).reshape((5,))
                hessian[j] += (1 - softmax[j]) * softmax[j] * np.dot(xi, xi.T)
            L -= (np.log(softmax[yi]))
        # scaling and regularization
        for j in range(class_num):
            hessian[j] /= n
            hessian[j] += 2 * lam * np.eye(5)
        grad /= n
        grad += 2 * lam * w
        L /= n
        obj_fun += L
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(obj_fun)))
        #obj_fun_history.append(np.asscalar(obj_fun))
        if (np.linalg.norm(grad) < 1e-6):
            break
        # update w
        for j in range(class_num):
            d = np.dot(np.linalg.inv(hessian[j]), grad[:,j])
            w[:,j] -= d
        step += 1
print("--- newton method ---")
print("step:" + str(step))
print(w)



# print("--- compare ---")
# tmp1 = arr_sgm[len(arr_sgm)-1]
# arr_sgm -= tmp1*np.ones((len(arr_sgm)))
# plt.plot(np.arange(0, len(arr_sgm), 1), arr_sgm)
# tmp2 = arr_newton[len(arr_newton)-1]
# arr_newton -= tmp2*np.ones((len(arr_newton)))
# plt.plot(np.arange(0, len(arr_newton), 1), arr_newton)
# #print(arr_sgm)
# #plt.plot(np.arange(0, len(arr_newton), 1), arr_newton)
# plt.show()
