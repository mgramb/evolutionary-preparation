from matplotlib import pyplot as plt
import numpy as np
import scipy as sc
import scipy.linalg


def matrix(x):
    # this is the matrix associated to perfect mixing (all options 1/3)
    return np.array([[5*x-31/6, 5/6-x, 29/6-5*x, x-7/6, 1/6, 1/6], [5/6-x, 5*x-31/6, x-7/6, 29/6-5*x, 1/6, 1/6],
                     [11/6-x, 5/6-x, x-7/6, x-7/6, -5/6, 1/6], [5/6-x, 11/6-x, x-7/6, x-7/6, 1/6, -5/6],
                     [5/6-x, 5/6-x, x-1/6, x-7/6, 1/6, 1/6], [5/6-x, 5/6-x, x-7/6, x-1/6, 1/6, 1/6]])


def evsorted(x):
    eigvalues, eigvectors = sc.linalg.eig(matrix(x))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d


t = np.arange(0.001, 1., 0.001)
a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []

for i in range(len(t)):
    a1.append(evsorted(t[i])[0])
    a2.append(evsorted(t[i])[1])
    a3.append(evsorted(t[i])[2])
    a4.append(evsorted(t[i])[3])
    a5.append(evsorted(t[i])[4])
    a6.append(evsorted(t[i])[5])

plt.figure(dpi=100)
plt.plot(t, a1, '--', t, a2, '--', t, a3, '--', t, a4, '--', t, a5, '--', t, a6, '--')
plt.xlabel("$\Delta$")
plt.ylabel("Real Parts of the Eigenvalues")
plt.show()


def matrix2(x):
    # this is the matrix associated to mixing between two options (in this case option 1 and 2)
    return np.array([[5 * x - 5, 1 - x, 5 - 5 * x, x - 1, 1 / 4, 1 / 4],
                     [1 - x, 5 * x - 5, x - 1, 5 - 5 * x, 1 / 4, 1 / 4],
                     [7/4 - x, 1/4 - x, x - 1, x - 1, -1/2, 1],
                     [1/4 - x, 7/4 - x, x - 1, x - 1, 1, -1/2],
                     [1 - x, 1 - x, x - 1 / 4, x - 7 / 4, -1 / 2, -1 / 2],
                     [1 - x, 1 - x, x - 7 / 4, x - 1 / 4, -1 / 2, -1 / 2]])


def evsorted2(x):
    eigvalues, eigvectors = sc.linalg.eig(matrix2(x))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d


b1 = []
b2 = []
b3 = []
b4 = []
b5 = []
b6 = []

for i in range(len(t)):
    b1.append(evsorted2(t[i])[0])
    b2.append(evsorted2(t[i])[1])
    b3.append(evsorted2(t[i])[2])
    b4.append(evsorted2(t[i])[3])
    b5.append(evsorted2(t[i])[4])
    b6.append(evsorted2(t[i])[5])


plt.figure(dpi=100)
plt.plot(t, b1, '--', t, b2, '--', t, b3, '--', t, b4, '--', t, b5, '--', t, b6, '--')
plt.xlabel("$\Delta$")
plt.ylabel("Real Parts of the Eigenvalues")
plt.show()


def matrix3(x):
    # this is the matrix associated to only playing and preparing for option 1
    return np.array([[5*x - 5, 1 - x, 5 - 5 * x, x - 1, 1 / 2, 1 / 2],
                     [1 - x, 5 * x - 5, x - 1, 5 - 5 * x, 1 / 2, 1 / 2],
                     [1 - x, 1 - x, x - 1, x - 1, 1 / 2, 1 / 2],
                     [1 - x, 1 - x, x - 1, x - 1, 1 / 2, 1 / 2],
                     [1 - x, 1 - x, x - 1, x - 1, -5 / 2, 1 / 2],
                     [1 - x, 1 - x, x - 1, x - 1, 1 / 2, -5 / 2]])


def evsorted3(x):
    eigvalues, eigvectors = sc.linalg.eig(matrix3(x))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d


c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []

for i in range(len(t)):
    c1.append(evsorted3(t[i])[0])
    c2.append(evsorted3(t[i])[1])
    c3.append(evsorted3(t[i])[2])
    c4.append(evsorted3(t[i])[3])
    c5.append(evsorted3(t[i])[4])
    c6.append(evsorted3(t[i])[5])


plt.figure(dpi=100)
plt.plot(t, c1, '--', t, c2, '--', t, c3, '--', t, c4, '--', t, c5, '--', t, c6, '--')
plt.xlabel("$\Delta$")
plt.ylabel("Real Parts of the Eigenvalues")
plt.show()


def matrix4(x):
    # this is the matrix associated to only playing option 1 and preparing for option 2
    return np.array([[5 * x - 5, 1 - x, 11 / 2 - 5 * x, x - 1 / 2, 0, 0],
                     [1 - x, 5 * x - 5, x - 1 / 2, 11 / 2 - 5 * x, 0, 0],
                     [1 - x, 1 - x, x - 7 / 2, x + 5 / 2, 0, 0],
                     [1 - x, 1 - x, x - 1 / 2, x - 13 / 2, 0, 0],
                     [1 - x, 1 - x, x - 1 / 2, x - 1 / 2, 3, 0],
                     [1 - x, 1 - x, x - 1 / 2, x - 1 / 2, -3, 0]])


def evsorted4(x):
    eigvalues, eigvectors = sc.linalg.eig(matrix4(x))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d


d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []

for i in range(len(t)):
    d1.append(evsorted4(t[i])[0])
    d2.append(evsorted4(t[i])[1])
    d3.append(evsorted4(t[i])[2])
    d4.append(evsorted4(t[i])[3])
    d5.append(evsorted4(t[i])[4])
    d6.append(evsorted4(t[i])[5])


plt.figure(dpi=100)
plt.plot(t, d1, '--', t, d2, '--', t, d3, '--', t, d4, '--', t, d5, '--', t, d6, '--')
plt.xlabel("$\Delta$")
plt.ylabel("Real Parts of the Eigenvalues")
plt.show()


def matrix5(x):
    # this is the matrix associated to mixing between option 1 and 2 with inconsistent preparation for option 3
    return np.array([[5 * x - 5, 1 - x, 23 / 4 - 5 * x, x - 1 / 4, -1/4, -1/4],
                     [1 - x, 5 * x - 5, x - 1 / 4, 23 / 4 - 5 * x, -1/4, -1/4],
                     [7/4 - x, 1/4 - x, x - 5 / 2, x - 5 / 2, -1, 1/2],
                     [1/4 - x, 7/4 - x, x - 5 / 2, x - 5 / 2, 1/2, -1],
                     [1 - x, 1 - x, x + 1 / 2, x - 1, 5/4, -1/4],
                     [1 - x, 1 - x, x - 1, x + 1 / 2, -1/4, 5/4]])


def evsorted5(x):
    eigvalues, eigvectors = sc.linalg.eig(matrix5(x))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d


e1 = []
e2 = []
e3 = []
e4 = []
e5 = []
e6 = []

for i in range(len(t)):
    e1.append(evsorted5(t[i])[0])
    e2.append(evsorted5(t[i])[1])
    e3.append(evsorted5(t[i])[2])
    e4.append(evsorted5(t[i])[3])
    e5.append(evsorted5(t[i])[4])
    e6.append(evsorted5(t[i])[5])


plt.figure(dpi=100)
plt.plot(t, e1, '--', t, e2, '--', t, e3, '--', t, e4, '--', t, e5, '--', t, e6, '--')
plt.xlabel("$\Delta$")
plt.ylabel("Real Parts of the Eigenvalues")
plt.show()


def matrix6(x, p):
    # this is the matrix associated to state (1,0,1,0,0,p2)
    return np.array([[5*x-5, 1-x, 6-5*x-p+p*p/2, x-3*p/2+p*p, (p-1)/2, 0],
                     [1-x, 5*x-5, x - p + p*p/2, 6-5*x-3*p/2+p*p, (p-1)/2, 0],
                     [1-x, 1-x, x-6+2*p+p*p/2, x-3+9*p/2+p*p, (p-1)/2, 0],
                     [1-x, 1-x, x-p+p*p/2, x-3-9*p/2+p*p, (p-1)/2, 0],
                     [1-x, 1-x, x-p+p*p/2, x-p/2-p*(1-p), (p+5)/2, 0],
                     [1-x, 1-x, x-p/2+5*p*(1-p)/2, x-3*p/2+p*p, (-5*p-1)/2, 0]])


def highestev6(x, p):
    eigvalues, eigvectors = sc.linalg.eig(matrix6(x, p))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d[0]


def lowestev6(x, p):
    eigvalues, eigvectors = sc.linalg.eig(matrix6(x, p))
    realev = eigvalues.real
    d = sorted(realev, reverse=True)
    return d[5]


def evforp(p):
    tdelta = np.arange(0.01, 1, 0.01)
    evlist = []
    for k in range(len(tdelta)):
        evlist.append(highestev6(tdelta[k], p))
    value = min(evlist)
    return value


def evlowforp(p):
    tdelta = np.arange(0.01, 1, 0.01)
    evlist = []
    for k in range(len(tdelta)):
        evlist.append(lowestev6(tdelta[k], p))
    value = max(evlist)
    return value


evplotlist = []
evlowplotlist = []
for i in range(len(t)):
    evplotlist.append(evforp(t[i]))
    evlowplotlist.append(evlowforp(t[i]))

plt.figure(dpi=100)
plt.plot(t, evplotlist, '--')
plt.xlabel("$p_2$")
plt.ylabel("Real Part of the Eigenvalue")
plt.show()


plt.figure(dpi=100)
plt.plot(t, evlowplotlist, '--')
# This plot was not included in the paper. For a fixed value of p_2, it shows the maximal value of the lowest real part
# of all eigenvalues across all \Delta in [0,1]
plt.xlabel("$p_2$")
plt.ylabel("Real Part of the Eigenvalue")
plt.show()

