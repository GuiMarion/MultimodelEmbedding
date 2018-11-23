'''
This file is used to plot all the functions we want in a same file.
'''
import matplotlib.pyplot as plt
import numpy as np
import valer as vlr

X = np.arange(0, 100, 1)
linear = np.copy(X)
plt.figure();
plt.plot(X, linear, label="y = x")
a,b = vlr.suite(100,1.05)
plt.plot(a, b, label="suite géométrique")
plt.legend()
plt.show()
