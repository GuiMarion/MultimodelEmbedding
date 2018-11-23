'''
This file is used to plot all the functions we want in a same file.
'''
import matplotlib.pyplot as plt
import numpy as np
    
def fact(n):
    if n <= 0 :
        return 1
    else :
        return n*fact(n-1)
   
def G(n):
    return fact(n-1)


plt.close('all')

n = len(X)
q = 0.96

linear = np.copy(X)

def Gamma(n):
    X = np.arange(0, n, 1.)
    return [G(x) for x in X]

n = 100
linear = np.log(Gamma(n))
X = np.arange(0, n, 1)

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
