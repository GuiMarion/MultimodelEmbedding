'''
This file is used to plot all the functions we want in a same file.
'''
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 100, 1)
linear = np.copy(X)
plt.figure();
plt.plot(X, linear, label="y = x")
plt.legend()
plt.show()
