import torch
import numpy as np
import time

def getKMax(k, L):
	temp = list(range(10))
	for i in range(len(L)):
		if L[i] > temp[0]:
			temp[0] = L[i]
			temp.sort()

	return temp

def old(k, L):
	L = np.asarray(L)
	L.sort()
	return L[-k:]

L = list(np.random.randint(100, size=1000000))

now = time.time()
getKMax(10, L)
print(time.time() - now)

now = time.time()
old(10, L)
print(time.time() - now)