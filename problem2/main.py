# requirement
import matplotlib.pyplot as plt

from problem2 import norm
import os
a = norm.seq(lam=2)
b = norm.seq(lam=4)
c = norm.seq(lam=6)
plt.semilogy(a, 'bs-', markersize=1, linewidth=0.5,label='lam=2')
plt.semilogy(b, 'rs-', markersize=1, linewidth=0.5,label='lam=4')
plt.semilogy(c, 'gs-', markersize=1, linewidth=0.5,label='lam=6')
dirname = "figures/"
plt.xlabel("step")
plt.ylabel("diff")
plt.legend()
os.makedirs(dirname, exist_ok=True)
filename = dirname + "plus.pdf"
plt.savefig(filename)
plt.show()