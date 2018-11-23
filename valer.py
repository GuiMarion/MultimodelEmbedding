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
