import matplotlib.pyplot as plt
from HartreeFock import HartreeFock

Ne = HartreeFock(10, [1,2,2], [0,0,1], [2,2,6])
Ar = HartreeFock(18, [1,2,2,3,3], [0,0,1,0,1], [2,2,6,2,6])
label = ['1s', '2s', '2p', '3s', '3p']

plt.subplot(2,1,1)

plt.axis([1e-3, 1e1, -2, 4])
for i,a in enumerate(Ne):
    plt.semilogx(a.r, a.P, label=label[i])

plt.legend(loc='upper right', labelspacing=0.1)
plt.ylabel('$P(r)$', fontsize=14)
plt.text(1.2e-3,3,'Ne', fontsize=14)

plt.subplot(2,1,2)

plt.axis([1e-3, 1e1, -2, 4])
for i,a in enumerate(Ar):
    plt.semilogx(a.r, a.P, label=label[i])

plt.legend(loc='upper right', labelspacing=0.1)
plt.ylabel('$P(r)$', fontsize=14)
plt.text(1.2e-3,3,'Ar', fontsize=14)

plt.xlabel('$r$  / Bohr radius', fontsize=14)
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
