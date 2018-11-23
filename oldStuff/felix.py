import matplotlib.pyplot as plt
import numpy as np

def fact(n):
    if n <= 0 :
        return 1
    else :
        return n*fact(n-1)

def G(n):
    return fact(n-1)

def felix(n) :
    X = np.arange(0, n, 1.)
    return [G(x) for x in X]
