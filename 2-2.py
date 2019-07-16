# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

def st_ops(mu, q):
  x_proj = np.zeros(mu.shape)
  for i in range(len(mu)):
    if mu[i] > q:
      x_proj[i] = mu[i] - q
    else:
      if np.abs(mu[i]) < q:
        x_proj[i] = 0
      else:
        x_proj[i] = mu[i] + q
  return x_proj


# we need to control this parameter to generate multiple figures
lamall = np.arange(0,5,0.1)

x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))

A = np.array([[3, 0.5],
              [0.5, 1]])
mu = np.array([[1],
               [2]])

x_init = np.array([[3],
                   [-1]])
L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

w_history=[]
for lam in lamall:
    xt = x_init
    for t in range(1000):
        grad = 2 * np.dot(A, xt - mu)
        xth = xt - 1 / L * grad
        xt = st_ops(xth, lam * 1 / L)
    w_history.append(xt.T)
w_history = np.vstack(w_history)
plt.plot(lamall,w_history[:, 0], label='w1')
plt.plot(lamall,w_history[:, 1], label='w2')
# plt.xlim(-1.5, 3)
# plt.ylim(-1.5, 3)
plt.show()