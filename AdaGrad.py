# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import st_func
# proximal gradient
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
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])
    eta0 = 300 / L

    x_history = []
    fvalues = []
    g_history = []
    delta = 0.02
    for t in range(100):
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt - mu)

        g_history.append(grad.flatten().tolist())
        ht = np.sqrt(np.sum(np.array(g_history) ** 2, axis=0).T) + delta
        ht = ht.reshape(2, 1)

        eta_t = eta0
        xth = xt - eta_t * (ht ** -1 * grad)
        ht_inv = ht ** -1
        xt = np.array([st_func.st_ops(xth[0], lam * eta_t * ht_inv[0]),
                       st_func.st_ops(xth[1], lam * eta_t * ht_inv[1])])

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)

    x_init = np.array([[3],
                       [-1]])
    return fvalues