'''
This file is used to plot all the functions we want in a same file.
'''

import fibo
import matplotlib.pyplot as plt
import numpy as np
import felix as fx

X = np.arange(0, 100, 1)
f = fibo.fib(1,1,100)
linear = np.copy(X)
plt.figure();
plt.plot(X, linear, label="y = x")
plt.plot(f, label="fibonacci")
plt.plot(X, fx.felix(100), label="Fonction Gamma")
plt.legend()
plt.show()


