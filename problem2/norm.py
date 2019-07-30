# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import os

def st_ops(mu, q):
  x_proj = np.zeros(mu.shape)
  for i in range(len(mu)):
    if mu[i] >= q:
      x_proj[i] = mu[i] - q
    else:
      if np.abs(mu[i]) < q:
        x_proj[i] = 0
      else:
        x_proj[i] = mu[i] + q
  return x_proj


def seq(lam):
    x_1 = np.arange(-1.5, 3, 0.01)
    x_2 = np.arange(-1.5, 3, 0.02)

    X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
    fValue = np.zeros((len(x_1), len(x_2)))

    A = np.array([[3, 0.5],
                  [0.5, 1]])
    mu = np.array([[1],
                   [2]])

    for i in range(len(x_1)):
        for j in range(len(x_2)):
            inr = np.vstack([x_1[i], x_2[j]])
            fValue[i, j] = np.dot(np.dot((inr - mu).T, A), (inr - mu)) + lam * (np.abs(x_1[i]) + np.abs(x_2[j]))

    # cvx
    w_lasso = cv.Variable((2, 1))
    obj_fn = cv.quad_form(w_lasso - mu, A) + lam * cv.norm(w_lasso, 1)
    objective = cv.Minimize(obj_fn)
    constraints = []
    prob = cv.Problem(objective, constraints)
    result = prob.solve(solver=cv.CVXOPT)
    w_lasso = w_lasso.value

    # plt.contour(X1, X2, fValue) #勾配線

    x_init = np.array([[3],
                       [-1]])
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    x_history = []
    xt = x_init
    for t in range(100):
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt - mu)
        xth = xt - 1 / L * grad
        xt = st_ops(xth, lam * 1 / L)

    x_history = np.vstack(x_history)

    tr = x_history.T
    x1 = tr[0]
    x2 = tr[1]
    x1 -= x1[len(x1) - 1]
    x2 -= x2[len(x1) - 1]
    x3 = np.abs(x1) + np.abs(x2)
    x3 = np.delete(x3, len(x3) - 1)
    print(x3)
    return x3