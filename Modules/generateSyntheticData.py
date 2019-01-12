import numpy as np
import time
import random

def generateSyntData(N, nbFonts, size1=(80,100), size2=(100, 128)):

	ret = []
	for n in range(N):
		tempX1 = np.random.random_sample(size1)
		tempX2 = []
		names = []
		for f in range(nbFonts):
			names.append(str(n)+"-font"+str(f))
			tempX2.append(np.random.random_sample(size2))

		ret.append((tempX1, tempX2, names))

	return ret

def getSmallSetofBatch(data, batchSize):
	# construct valid batchs from data, is effiencient if len(data) <=32

	numberofData = len(data)*len(data[0][2])
	numberofBatches = numberofData // batchSize
	batches = []
	empty = []

	for b in range(numberofBatches):

		X1 = []
		X2 = []
		L1 = []
		L2 = []
		unseenX = np.delete(np.arange(len(data)), empty)
		for i in range(batchSize):
			k1 = np.random.randint(len(unseenX))
			k = unseenX[k1]
			X1.append(data[k][0])
			k2 = np.random.randint(len(data[k][1]))
			X2.append(data[k][1][k2])
			L1.append(data[k][2][0][:data[k][2][0].find("-")])
			L2.append(data[k][2][k2])
			# We delete the indice we just saw
			unseenX = np.concatenate((unseenX[:k1], unseenX[k1+1:]), axis=None)
			# We also delete the spectrum we used
			del data[k][1][k2]
			del data[k][2][k2]

			if len(data[k][2]) == 0 :
				empty.append(k)

		# We shuffle X2 in order to don't the make matching indices in each batch
		# We need to shuffle L2 the same way in order to get the right names
		c = list(zip(X2, L2))
		random.shuffle(c)
		X2, L2 = zip(*c)

		batches.append((X1, X2, L1, L2))


	return batches

def getBatches(data, batchSize):
	# return efficiently valid batches from data if len(data) > 32
	random.shuffle(data)

	batches = []
	for i in range(len(data) // batchSize):
		batches.extend(getSmallSetofBatch(data[i*batchSize : (i+1)*batchSize], batchSize))

	return batches



def testMyBatchFunction(N, batchSize):

	data = generateSyntData(N, 4)
	print("____ Executing the function")
	start_time = time.time()
	batchSet = getBatches(data, batchSize)
	print("--- %s seconds ---" % (time.time() - start_time))

	print("____ Testing the results")

	if isBatchValid(batchSet, batchSize):
		print("Congrats, your function is OK")
		return True
	else:
		print("Unfertunatly your functions doesn't return the right results ...")
		return False


def isBatchValid(batchSet, batchSize):
		
	if batchSet == "":
		print("Seems that you need to code your function")
		return False

	if len(batchSet) == 0:
		print("Your batchSet is empty ... ")
		return False

	waveFormSeen = []

	for batch in batchSet:
		if len(batch[0]) != batchSize:
			print("One of your batchs doesn't have the right length")
			return False


		X1, X2, L1, L2 = batch

		# for every element in the batch
		for i in range(len(X1)):
			matches = 0
			# we count the number of matches it has
			for j in range(len(X1)):
				if L1[i] == L2[j][:L2[j].find("-")]:
					matches += 1
					if L2[j] in waveFormSeen:
						print("Seems that a sectrum has been seen twice.")
						return False
					waveFormSeen.append(L2[j])
				# this number should be 1, so we return false if it's more 
				if matches > 1:
					print(L1[i], "seems to appear twice")
					return False
			if matches != 1:
				return False

	return True


