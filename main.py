# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

import Adam
import AdaDelta
import AdaGrad
import RMSProp
import Nadam

a = AdaGrad.opt()
b = Adam.opt()
c = RMSProp.opt()
d = AdaDelta.opt()
e = Nadam.opt()
plt.semilogy(a, 'bs-', markersize=1, linewidth=0.5,label='AdaGrad')
plt.semilogy(b, 'r-', markersize=1, linewidth=0.5,label='Adam')
plt.semilogy(c, 'g-', markersize=1, linewidth=0.5,label='RMSProp')
plt.semilogy(d, 'k-', markersize=1, linewidth=0.5,label='AdaDelta')
plt.semilogy(e, 'y-', markersize=1, linewidth=0.5,label='Nadam')
plt.savefig("compare7.pdf")
plt.legend()
plt.show()

