import numpy as np

def generateSyntData(N, nbFonts, size1=(80,100), size2=(100, 128)):
	X1 = []
	X2 = []
	L = []
	for n in range(N):
		tempX1 = np.random.random_sample(size1)
		for f in range(nbFonts):
			tempX2 = np.random.random_sample(size2)
			X1.append(tempX1)
			X2.append(tempX2)
			L.append(str(n)+"_font"+str(f))

	return (X1, X2, L)
