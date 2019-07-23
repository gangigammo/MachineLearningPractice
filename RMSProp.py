# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import st_func

def opt():
    x_1 = np.arange(-1.5, 3, 0.01)
    x_2 = np.arange(-1.5, 3, 0.02)

    X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
    fValue = np.zeros((len(x_1), len(x_2)))

    A = np.array([[250, 15],
                  [15, 4]])
    mu = np.array([[1],
                   [2]])
    lam = 0.89

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

    x_init = np.array([[3],
                       [-1]])
    xt = x_init

    b1 = 0.7
    b2 = 0.99999
    ee = 1.0e-8
    aa = 0.2

    x_history = []
    fvalues = []
    g_history = []


    hh = np.zeros((2, 1))
    ler = 0.1
    alpha = 0.95
    e = 1e-6

    for t in range(1, 101):
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt - mu)

        hh = alpha * hh + (1-alpha) * (grad*grad)
        g_history.append(grad.T)
        tmpler = ler/(np.sqrt(hh)+e)
        xth = xt - grad * tmpler

        xt = np.array([st_func.st_ops(xth[0], lam * tmpler[0]),
                       st_func.st_ops(xth[1], lam * tmpler[1])])

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return fvalues
