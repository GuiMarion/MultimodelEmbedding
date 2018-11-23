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

def Gamma(n):
    X = np.arange(0, n, 1.)
    return [G(x) for x in X]

n = 100
linear = np.log(Gamma(n))
X = np.arange(0, n, 1)

plt.plot(X, linear, label="y = x")
plt.legend()
plt.show()
