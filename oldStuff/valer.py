# Babilobobop
# Bababoooooooo

import numpy as np

def geo(i,q):
    if i == 0:
        return(1)
    return geo(i-1,q)*q

def suite(n,q):
    u = np.ones(n)
    X = np.arange(0, n, 1)
    for i in range(n):
        u[i] = geo(i,q)
    return X,u
