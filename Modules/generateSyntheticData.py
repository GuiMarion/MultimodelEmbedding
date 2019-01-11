import numpy as np
import time

def generateSyntData(N, nbFonts, size1=(80,100), size2=(100, 128)):

	ret = []
	for n in range(N):
		tempX1 = np.random.random_sample(size1)
		for f in range(nbFonts):
			tempX2 = np.random.random_sample(size2)
			name = str(n)+"_font"+str(f)

			ret.append((tempX1, tempX2, name))

	return ret

def MYFUNCTION(dataset, batchSize):
	pass

def testMyBatchFunction(batch, batchSize, data):
	print("____ Executing the function")
	start_time = time.time()
	batchSet = MYFUNCTION()
	print("--- %s seconds ---" % (time.time() - start_time))

	print("____ Testing the results")

	if isBatchValid(batchSet, batchSize):
		print("Congrats, the function is OK")
	else:
		print("Unfertunatly your functions doesn't return the right results ...")


def isBatchValid(batchSet, batchSize):
	
	for batch in batchSet:
		if len(batch) != batchSize:
			return False

		X1, X2, L1, L2 = batch

		# for every element in the batch
		for i in range(len(X1)):
			matches = 0
			# we count the number of mateches it has
			for j in range(len(X1)):
				if L[i] == L[j][:L[j].rfind("-")]:
					matches += 1
				# this number should be 1, so we return false if it's more 
				if matches > 1:
					return False
			if matches != 1:
				return False

	return True


