import numpy as np
import time
import random

def generateSyntData(N, nbFonts, size1=(80,100), size2=(100, 128)):
	"""Generates synthetic data in order to test the functions and modules of the project.
	
	Returns a list containing random numpy arrays that have the same shape that the 
	multimodal data in the dataset, and are ordered by couples with a given name.
	Notice that returned data simulates the fact that several audio (CQT or STFT) 
	snippets (with different fonts) can be associated with one midi (pianoroll) 
	snippet.
	
	Parameters
	----------
	N : int
		Number of synthetical midi snippets computed.
	nbFonts : int
		Number of audio snippets associated with each midi snippet. Notice that
		if nbFonts is more than one, several couples of data associated with a
		same midi data will be computed.
	size1 : list of int, optional
		Shape of the synthetical midi snippet.
	size2 : list of int, optional
		Shape of the synthetical audio snippet.
		
	Returns
	-------
	ret : list
		List containing the synthetic data.
	"""
	
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
	""" Constructs batches from data that can be sent in the neural network.
	
	Returns a list of batches from the input data. Each batch contains only one
	matching couple. This function is only efficient if the length 
	of the input data is inferior or equal to 32.
	
	Parameters
	----------
	data : list
		Input data. Contains couple of snippets, associated with a given name.
	batchSize : int
		Size of the batches to compute.
	
	Returns
	-------
	batches : list
		List of batches to feed the network.
	"""
	
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
	""" Returns valid batches from data. Efficient if len(data) > 32.
	
	Notice that it calls getSmallSetofBatch.
	
	Parameters
	----------
	data : list
		Input data. Contains couple of snippets, associated with a given
		name.
	batchSize : int
		Size of the batches to compute.
	
	Returns
	-------
	batches : list
		List of batches to feed the network.
	"""
	
	random.shuffle(data)

	batches = []
	for i in range(len(data) // batchSize):
		batches.extend(getSmallSetofBatch(data[i*batchSize : (i+1)*batchSize], batchSize))

	return batches


def testMyBatchFunction(N, batchSize):
	"""Tests the functions in this module.
	
	Returns a boolean that indicates if the functions are working properly.
	
	Parameters
	----------
	N : int
		Number of midi snippets to compute with generateSynthData. Notice that each
		snippet will correspond to 4 different audio snippets.
	batchSize : int
		Size of the batches to compute.
	
	Returns
	-------
	bool
		True if everything is working properly. False otherwise.
	"""

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
		print("Unfortunately your functions don't return the right results ...")
		return False


def isBatchValid(batchSet, batchSize):
	""" Checks if a given set of batches is valid to be sent in the neural network.
	
	Notice that, to be valid, each batch on the set must have the same size, and
	should contain only one matching midi/audio couple of snippets. Returns a 
	boolean that indicate if the batch set is valid or not.
	
	Parameters
	----------
	batchSet : list
		Contains the set of batchs to be checked.
	batchSize : int
		Length of the batchs in the set.
		
	Returns
	-------
	bool
		True if the batch set is valid. False otherwise.
	"""
	
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


