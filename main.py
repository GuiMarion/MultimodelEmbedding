'''
This file is used to plot all the functions we want in a same file.
'''
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')


X = np.arange(0, 100, 1)
n = len(X)
q = 0.96

linear = np.copy(X)

def geo(i):
    if i == 0:
        return(1)
    return geo(i-1)*q

def suite(n):
    u = np.ones(n)
    X = np.arange(0, n, 1)
    for i in range(n):
        u[i] = geo(i)
    return X,u
    

plt.figure();
plt.plot(X, linear, label="y = x")
plt.legend()
plt.show()

a,b = suite(n)

plt.figure();
plt.plot(a,b,label='Suite géométrique')
plt.show()
