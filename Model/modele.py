
from Modules import dataBase
from Model import network
from Modules import waveForm
from Modules import score

import numpy as np
import torch
import pickle
from tqdm import tqdm
import statistics
import operator
import os

try:
	import matplotlib.pyplot as plt
	plot = True
except ImportError:
	plot = False


class Modele():
	"""Defines the general model, with two convolutional neural networks.
	
	This class defines the model that learns simultaneously from piano rolls and 
	spectrograms, embedding both in a common latent space.
	
	Attributes
	----------
	GPU : bool
		True if the GPU of the computer is used, False otherwise.
	outPath : str
		The path in which the parameters are to be stored.
	model1
		The first neural network.
	model2
		The second neural network.
	batch_size : int
		Size of the input data batches.
	databasePath : str
		The folder in which the database is stored.
	losses : :obj: 'list' of :obj: 'float'
		History of the losses over time (for plotting purpose).
	losses_test : :obj: 'list' of :obj: 'float'
		History of the losses over test data.
	"""
	
	def __init__(self, databasePath=None, batch_size=32, gpu=None, outPath="/fast-1/guilhem/params/"):

		if gpu is None:
			self.GPU = False
		else :
			self.GPU = True

		self.outPath = outPath

		self.model1 = network.Net()
		self.model2 = network.Net()

		if self.GPU:
			torch.backends.cudnn.benchmark = True
			torch.cuda.set_device(gpu)
			self.model1 = self.model1.cuda()
			self.model2 = self.model2.cuda()

		self.batch_size = batch_size

		self.databasePath = databasePath

		self.loadBatch()

		self.losses = []
		self.losses_test = []

		self.dico = {}

		self.s = torch.nn.CosineSimilarity(dim=0)


		# We don't store loss greater than that
		self.lastloss = 50

	def loadBatch(self):
		"""Loads mini batches from database."""

		if self.databasePath is None:

			print("____ Loading batch at random: ")

			self.X1 = [torch.randn(self.batch_size, 1, 80, 100)-1 for i in range(self.batch_size)]
			self.X2 = [torch.randn(self.batch_size, 1, 80, 100)-1 for i in range(self.batch_size)]
			self.L1 = [str(i) for i in range(self.batch_size)]
			self.L2 = [str(j) for j in range(self.batch_size)]

		else:
			print("____ Loading batches from file")

			## Construct and save database
			D = dataBase.dataBase(outPath=self.outPath)
			D.constructDatabase(self.databasePath)
			self.batches = D.getBatches(self.batch_size)

			self.testBatches = D.getTestSet(self.batch_size)

			self.validationBatches = D.getValidationSet(self.batch_size)

			print("We have", len(self.batches), "batches for the training part.")



	def TestEval(self, batches):
		"""Evaluation fonction that returns the meaned loss for all batches.
		
		Parameters
		----------
		batches : :obj: 'list'
			List of all data batches.
			
		Returns
		-------
		float
			Meaned loss for all batches.
		"""
		
		if len(batches) == 0:
			return -1

		loss = 0

		for batch in batches:

			N1 = np.array(batch[0]).astype(float)
			N1 = N1.reshape(self.batch_size, 1, N1.shape[1], N1.shape[2])
			X1 = torch.autograd.Variable(torch.FloatTensor(N1), requires_grad=False)
			if self.GPU:
				X1 = X1.cuda()

			N2 = np.array(batch[1]).astype(float)
			N2 = N2.reshape(self.batch_size, 1, N2.shape[1], N2.shape[2])
			X2 = torch.autograd.Variable(torch.FloatTensor(N2), requires_grad=False)
			if self.GPU:
				X2 = X2.cuda()

			y_pred1 = self.model1.forward(X1)
			y_pred2 = self.model2.forward(X2)

			L1 = batch[2]
			L2 = batch[3]
			indices = batch[4]

			# Compute and print loss
			loss += self.myloss((y_pred1, y_pred2, L1, L2, indices)).item()

		return loss/len(batches)

	def save_weights(self):
		""" Saves the weights of the models in two separates files."""
		
		print("____ Saving the models.")

		if not os.path.exists(self.outPath + "params/"):
			os.makedirs(self.outPath + "params/")

		torch.save(self.model1.cpu(), self.outPath + "params/model1.data")
		torch.save(self.model2.cpu(), self.outPath + "params/model2.data")

		if self.GPU:
			self.model1 = self.model1.cuda()
			self.model2 = self.model2.cuda()

	def plot_losses(self):
		"""Plots the losses over time."""
		
		if plot == True:
			loss, = plt.plot(np.array(self.losses), label='Loss on training')
			lossTest, = plt.plot(np.array(self.losses_test), label='Loss on test')
			plt.legend(handles=[loss, lossTest])
			plt.show()
		else:
			print("Impossible to plot, tkinter not available.")


	def myloss(self, batch, alpha=0.7):
		'''Loss to be used for training and evaluation.
		
		Same loss function that the one used in Dorfer et al. [2018].
		Pairwise hinge loss, using cosine similarity.
		
		Parameters
		----------
		batch
			Data batch over which we calculate the loss.
		alpha : float, optional
			Hinge loss parameter (defaults to 0.7).
		Returns
		-------
		rank : float
			Optimized pairwise ranking objective (the "hinge loss").
		'''

		X1, X2, L1, L2, indices = batch

		rank = 0

		for x in range(len(X1)):
			for y in range(len(X2)):
				if y != x:
					rank += max(0, alpha - self.s(X1[indices[x]], X2[x]) + self.s(X1[indices[x]], X2[y]))
					if L1[indices[x]] != L2[x][:L2[x].find("-")]:
						print("ERREUR")
						print("Should match", L1[indices[x]], L2[x][:L2[x].find("-")])

					if L1[indices[x]] == L2[y][:L2[x].find("-")]:
						print("ERREUR")
						print("Should not match", L1[indices[x]], L2[y][:L2[x].find("-")])

		return rank

	def constructDict(self):
		'''Constructs a dictionary allowing to match coordinates in the latent space.
		
		With a name of music piece and another one that match names with pianoroll matrices.
		'''

		print("____ Constructing the dictionary")

		dico = {}
		self.model1.eval()

		rollsFromName = {}


		for batch in tqdm(self.batches):

			N1 = np.array(batch[0]).astype(float)
			N1 = N1.reshape(self.batch_size, 1, N1.shape[1], N1.shape[2])
			X1 = torch.FloatTensor(N1)

			if self.GPU:
				X1 = X1.cuda()

			Y = self.model1.forward(X1).data
			for i in range(len(batch[2])):
				dico[Y[i]] = batch[2][i]

				rollsFromName[batch[2][i]] = batch[0][i]

		pickle.dump(dico, open( self.outPath + "dico.data", "wb" ) )

		self.dico = dico
		self.rollsFromName = rollsFromName



	def learn(self, EPOCHS, learning_rate=1e-7, momentum=0.9):
		'''Learn method of the model, that trains the model on self.batches.
		
		Parameters
		----------
		EPOCHS : int
			Number of passes through the model.
		learning_rate : float, optional
			Learning rate of the neural networks.
		momentum : float, optional
			Parameter for the pytorch SGD optimizer.
		'''

		print("_____ Training")
		parameters = [p for p in self.model1.parameters()] + [p for p in self.model2.parameters()]

		optimizer = torch.optim.Adam(parameters, lr=learning_rate) ## if you can use +
		for t in range(EPOCHS):
			# Make learn the two models with respects to x and y

			for batch in tqdm(self.batches):
				N1 = np.array(batch[0]).astype(float)
				N1 = N1.reshape(self.batch_size, 1, N1.shape[1], N1.shape[2])
				X1 = torch.autograd.Variable(torch.FloatTensor(N1), requires_grad=True)
				if self.GPU:
					X1 = X1.cuda()

				N2 = np.array(batch[1]).astype(float)
				N2 = N2.reshape(self.batch_size, 1, N2.shape[1], N2.shape[2])
				X2 = torch.autograd.Variable(torch.FloatTensor(N2), requires_grad=True)
				if self.GPU:
					X2 = X2.cuda()

				y_pred1 = self.model1.forward(X1)
				y_pred2 = self.model2.forward(X2)

				L1 = batch[2]
				L2 = batch[3]
				indices = batch[4]

				# Compute the loss
				loss = self.myloss((y_pred1, y_pred2, L1, L2, indices))
				#print(t, loss.item())

				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


			# appending losses
			self.losses.append(float(loss.item()))
			self.losses_test.append(self.TestEval(self.testBatches))

			print()
			print("________ EPOCH ", t)
			print()
			print("____ Train Loss:", loss.item())
			print("____ Test Loss:", self.losses_test[t])

			# Store the model having the lowest loss on test set
			if self.losses_test[t] < self.lastloss:
				self.save_weights()
				self.lastloss = self.losses_test[t]


		self.plot_losses()

		try:
			self.model1 = torch.load(self.outPath + "params/model1.data")
			self.model2 = torch.load(self.outPath +  "params/model2.data")
		except FileNotFoundError:
			print("No model have been saved, the reason seems to be that no model was good enough, \
				try to train it on more EPOCHS.")

		if self.GPU:
			self.model1 = self.model1.cuda()
			self.model2 = self.model2.cuda()

		print()
		print("Test Loss for the best trained model:", self.TestEval(self.testBatches))

		print()
		self.constructDict()

		print()
		score = self.testBenchmark()

		print("Benchmark score:", score)
		print()

		print(self.losses)
		print(self.losses_test)

		pickle.dump((self.losses, self.losses_test, self.TestEval(self.testBatches), score), open( self.outPath + "losses.data", "wb" ) )


	'''
	Part for the evaluation of the model

	'''


	def constructDictForTest(self):
		'''Construct a dictionary allowing to match coordinates in the latent space.
		
		With a name of music piece and another one that match names with pianoroll matrices.
		This function only does this with the validation dataset.
		'''

		print("____ Constructing the dictionary")

		dico = {}
		self.model1.eval()

		rollsFromName = {}


		for batch in tqdm(self.validationBatches):

			N1 = np.array(batch[0]).astype(float)
			N1 = N1.reshape(self.batch_size, 1, N1.shape[1], N1.shape[2])
			X1 = torch.FloatTensor(N1)

			if self.GPU:
				X1 = X1.cuda()

			Y = self.model1.forward(X1).data
			for i in range(len(batch[2])):
				dico[Y[i]] = batch[2][i]

				rollsFromName[batch[2][i]] = batch[0][i]

		self.dicoEval = dico
		self.rollsFromNameEval = rollsFromName


	def nearestNeighborEval(self, wavePosition):
		'''Searchs for the nearest neighbor in the evaluation set.
		
		Parameters
		----------
		wavePosition
			Coordinates of the wave object in the latent space.
		
		Returns
		-------
		name : str
			The name of the nearest neighbor.
		'''

		dist = 1000000
		name = ""
		for key in self.dicoEval:
			tmp_dist = self.s(key, wavePosition[0])
			if  tmp_dist < dist:
				dist = tmp_dist
				name = self.dicoEval[key]

		return name


	def nearestNeighbor(self, wavePosition):
		'''Searchs for the nearest neighbor in the training set.
		
		Parameters
		----------
		wavePosition
			Coordinates of the wave object in the latent space.
		Returns
		-------
		name : str
			The name of the nearest neighbor.
		'''

		dist = 1000000
		name = ""
		for key in self.dico:
			tmp_dist = self.s(key, wavePosition[0])
			if  tmp_dist < dist:
				dist = tmp_dist
				name = self.dico[key]

		return name

	def pianorollDistance(self, p1, p2):
		'''Computes a distance between two pianorolls.
		
		Computes the distance using pairwise comparison only on non-zero elements
		in order to avoid false high similarity due to sparsity.
		
		Parameters
		----------
		p1
			First pianoroll.
		p2
			Second pianoroll.
		
		Returns
		-------
		float
			The distance between the two pianorolls.
		'''

		TP = 0.
		false = 0.

		if np.count_nonzero(p1) == 0:
			TP = 1

		for i in range(len(p1)):
			for j in range(len(p1[i])):
				if p1[i][j] == p2[i][j]:
					if p1[i][j] == 1:
						TP +=1 
				else:
					false += 1

		return TP / false


	def testBenchmarkEval(self):
		'''Processes an evaluation test we designed.
		
		It returns the meaned quantity of non-zeros pianoroll elements
		that have been correctly predicted. Only on the evaluation set.
		
		Returns
		-------
		score : float
			Meaned quantity of correctly predicted non-zeros pianoroll elements.
		'''
		
		score = 0.0
		for batch in self.testBatches:
			for i in range(len(batch[1])):
				N2 = np.array(batch[1][i]).astype(float)
				N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
				X2 = torch.FloatTensor(N2)
				if self.GPU:
					X2 = X2.cuda()

				Y2 = self.model2.forward(X2).data

				pianoroll1 = batch[0][batch[4][i]]
				pianoroll2 = self.rollsFromNameEval[self.nearestNeighborEval(Y2)]

				score += self.pianorollDistance(pianoroll1, pianoroll2)

		score /= (len(self.testBatches) * len(self.testBatches[0][2]))

		return score

	def testBenchmark(self):
		'''Processes an evaluation test we designed.
		
		It returns the meaned quantity of non-zeros pianoroll elements
		that have been correctly predicted. Only on the training set.
		
		Returns
		-------
		score : float
			Meaned quantity of correctly predicted non-zeros pianoroll elements.
		'''

		score = 0
		for batch in self.testBatches:
			for i in range(len(batch[1])):
				N2 = np.array(batch[1][i]).astype(float)
				N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
				X2 = torch.FloatTensor(N2)
				if self.GPU:
					X2 = X2.cuda()

				Y2 = self.model2.forward(X2).data

				pianoroll1 = batch[0][batch[4][i]]
				pianoroll2 = self.rollsFromName[self.nearestNeighbor(Y2)]

				score += self.pianorollDistance(pianoroll1, pianoroll2)

		score /= (len(self.testBatches) * len(self.testBatches[0][2]))

		return score

	def RecallK(self, k, embedded, expected):
		'''Computes the Recall@k (R@k) for one element.
		
		Parameters
		----------
		k : int
			Rank of the R@k we want to compute.
		embedded
			Coordinates of the element in the latent space.
		expected
			Original piano roll.
		'''

		recall = {}
		for key in self.dicoEval:
			recall[self.s(key, embedded[0])] = self.dicoEval[key]

		keylist = list(recall.keys())
		keylist.sort()

		for i in range(k):
			if np.array_equal(self.rollsFromNameEval[recall[keylist[i]]], expected):
				return True
		return False

	def getRecallK(self, k):
		'''Returns the meaned Recall@k (R@k) for all of the validation set.
		
		Parameters
		----------
		k : int
			Rank of the R@k we want to compute.
		
		Returns
		-------
		score : float
			Meaned R@k for the validation set.
		'''

		score = 0
		for batch in self.validationBatches:
			for i in range(len(batch[1])):
				N2 = np.array(batch[1][i]).astype(float)
				N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
				X2 = torch.FloatTensor(N2)
				if self.GPU:
					X2 = X2.cuda()

				Y2 = self.model2.forward(X2).data

				pianoroll1 = batch[0][batch[4][i]]
				
				score += self.RecallK(k, Y2, pianoroll1)

		score /= (len(self.validationBatches) * len(self.validationBatches[0][2]))

		return score

	def MRR(self, embedded, expected):
		'''Computes the Mean Reciprocal Rank (MRR) for one element.

		Parameters
		----------
		embedded
			Coordinates of the element in the latent space.
		expected
			Original piano roll.
		Returns
		-------
		float
			Estimated MRR for the element.
		'''

		recall = {}
		for key in self.dicoEval:
			recall[self.s(key, embedded[0])] = self.dicoEval[key]

		keylist = list(recall.keys())
		keylist.sort()

		for i in range(len(keylist)):
			if np.array_equal(self.rollsFromNameEval[recall[keylist[i]]], expected):
				return 1/(i+1)

	def getMRR(self):
		'''Returns the Mean Reciprocical Rank (MRR) for all of the validation set.
		
		Returns
		-------
		score : float
			Meaned MRR for the validation set.
		'''
		
		score = 0
		for batch in self.validationBatches:
			for i in range(len(batch[1])):
				N2 = np.array(batch[1][i]).astype(float)
				N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
				X2 = torch.FloatTensor(N2)
				if self.GPU:
					X2 = X2.cuda()

				Y2 = self.model2.forward(X2).data

				pianoroll1 = batch[0][batch[4][i]]
				
				score += self.MRR(Y2, pianoroll1)

		score /= (len(self.validationBatches) * len(self.validationBatches[0][2]))

		return score

	def MR(self, embedded, expected):
		'''Computes the Median Rank (MR) for one element.

		Parameters
		----------
		embedded
			Coordinates of the element in the latent space.
		expected
			Original piano roll.
		Returns
		-------
		int
			Estimated MR for the element.
		'''

		recall = {}
		for key in self.dicoEval:
			recall[self.s(key, embedded[0])] = self.dicoEval[key]

		keylist = list(recall.keys())
		keylist.sort()

		for i in range(len(keylist)):
			if np.array_equal(self.rollsFromNameEval[recall[keylist[i]]], expected):
				return i+1

	def getMR(self):
		'''Returns the Median Rank (MR) for all of the validation set.
		
		Returns
		-------
		score : float
			Median score for the validation set.
		'''

		scores = []
		for batch in self.validationBatches:
			for i in range(len(batch[1])):
				N2 = np.array(batch[1][i]).astype(float)
				N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
				X2 = torch.FloatTensor(N2)
				if self.GPU:
					X2 = X2.cuda()

				Y2 = self.model2.forward(X2).data

				pianoroll1 = batch[0][batch[4][i]]
				
				scores.append(self.MR(Y2, pianoroll1))

		return statistics.median(scores)

	def whatIsThisSong(self, file):
		'''Returns the name of the predicted song.
		
		Finds the nearest neighbors among snippets in the latent space,
		and returns the song whose name occurs the most.
		
		Parameters
		----------
		file : str
			Name of the audio file to test.
			
		Returns
		-------
		str
			Name of the predicted song.
		'''

		self.model2.eval()

		windowSize = 4 * 24 # 24 is the quantization we used to train the model
		STEP = 2
		r = 1.7945472249269718 # is the ratio between CQT and pianoroll size, is constant
		wave = waveForm.waveForm(file).getCQT()

		counter = {}

		for i in range(len(wave) - windowSize):
			snippet = wave[:,round(i*STEP*r) : round(i*STEP*r) + round(windowSize*r)]
			N2 = np.array(snippet).astype(float)
			N2 = N2.reshape(1, 1, N2.shape[0], N2.shape[1])
			X2 = torch.FloatTensor(N2)   
			embedded = self.model2.forward(X2)
			name = self.nearestNeighbor(embedded)
			if name in counter:
				counter[name] += 1
			else:
				counter[name] = 1

		sorted_counter = sorted(counter.items(), key=operator.itemgetter(1))

		return sorted_counter[-1][0]

	def addToDictionary(self, file, computeSound=True, pathTemp="/fast/guilhem/"):
		'''Adds all pianoroll snippets extracted from a given file.
		
		Parameters
		----------
		file : str
			Name of the file from which we extract the snippets.
		computeSound : bool, optional
			If True, creates a .wav file to inject in the latent space.
		pathTemp : str, optional
			The folder in which we save the temporary WaveForm files.
		'''

		windowSize = 4
		STEP = 2
		snippets = []

		self.model1.eval()

		s = score.score(file, outPath=pathTemp)
		if computeSound is True:
			file = file[file.rfind("/")+1:]
			file = file[:file.rfind(".")]
			s.toWaveForm().save(pathTemp +  file + ".wav")

		windowSize *= s.quantization
		N = s.length * s.quantization

		#for i in range((N-windowSize)//STEP):
		for i in range(10):
			tmpPart1 = s.extractPart(i*STEP, i*STEP+windowSize)
			N1 = np.array(tmpPart1.getPianoRoll()).astype(float)
			N1 = N1.reshape(1, 1, N1.shape[0], N1.shape[1])
			X1 = torch.FloatTensor(N1)  
			embedded = self.model1.forward(X1)[0]

			self.dico[embedded] = tmpPart1.name


