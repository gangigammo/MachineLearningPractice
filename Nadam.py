# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
# proximal gradient

# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import st_func

def opt():
    A = np.array([[250, 15],
                  [15, 4]])
    mu = np.array([[1],
                   [2]])
    lam = 0.89

    x_init = np.array([[3],
                       [-1]])
    xt = x_init

    x_history = []
    fvalues = []
    g_history = []
    m_t = np.zeros((2, 1))
    v_t = np.zeros((2, 1))
    mu_p = 0.5
    eps = 1e-16
    ups = 0.9999999
    lr = 0.2

    for t in range(1, 101):
        x_history.append(xt.T)
        # mu_t < - mu * (1 - mu ^ (t - 1)) / (1 - mu ^ t)
        # mu_tp1 < - mu * (1 - mu ^ t) / (1 - mu ^ (t + 1))
        # ups_t < - ups * (1 - ups ^ (t - 1)) / (1 - ups ^ t)
        # m_t < - mu_t * m_tm1 + (1 - mu_t) * g_t
        # mbar_t < - mu_tp1 * m_t + (1 - mu_t) * g_t(Nesterov)
        # v_t < - ups_t * v_tm1 + (1 - ups_t) * g_t ** 2
        # vbar_t < - sqrt(v_t) + eps
        # s_t < - lr * mbar_t / vbar_t
        # x_t < - x_t - s_t
        g_t = 2 * np.dot(A, xt - mu)
        mu_t = mu_p * (1 - mu_p ** (t - 1)) / (1 - mu_p ** t)
        mu_tp1 = mu_p * (1 - mu_p ** t) / (1 - mu_p ** (t + 1))
        ups_t = ups * (1 - ups ** (t - 1)) / (1 - ups ** t)
        m_t = mu_t * m_t + (1 - mu_t) * g_t
        mbar_t = mu_tp1 * m_t + (1 - mu_t) * g_t
        v_t = ups_t * v_t + (1 - ups_t) * g_t ** 2

        vbar_t = np.sqrt(v_t) + eps

        s_t = lr * mbar_t / vbar_t
        rateProx = lr / vbar_t
        xth = xt - s_t

        xt = np.array([st_func.st_ops(xth[0], lam * rateProx[0]),
                       st_func.st_ops(xth[1], lam * rateProx[1])])

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return fvalues
