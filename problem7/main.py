# requirement
import matplotlib.pyplot as plt

from problem7 import AdaDelta, AdaGrad, Adam, Nadam, RMSProp
import os
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
dirname = "figures/"
plt.xlabel("step")
plt.ylabel("diff")
plt.legend()
os.makedirs(dirname, exist_ok=True)
filename = dirname + "compare7.pdf"
plt.savefig(filename)
plt.show()

