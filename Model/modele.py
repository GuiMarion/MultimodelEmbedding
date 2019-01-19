
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
			D = dataBase.dataBase()
			D.constructDatabase(self.databasePath)
			self.batches = D.getBatches(self.batch_size)

			self.testBatches = D.getTestSet(self.batch_size)

			self.validationBatches = D.getValidationSet(self.batch_size)

			print("We have", len(self.batches), "batches for the training part.")



	def loss_test(self, y_pred1, y_pred2):
		"""Computes the loss for the final test part.
		
		Parameters
		----------
		y_pred1
			First prediction to compare (pytorch tensor).
		y_pred2
			Second prediction to compare (pytorch tensor).
		
		Returns
		-------
		loss : float
			The distance between the two tensors.
		"""
		
		# use the MSE for now
		if len(y_pred1) != self.model1.dim_latent and len(y_pred2) != self.model2.dim_latent:
			raise RuntimeError("y_pred1 and y_pred2 doesn't have same shape for test.")

		loss = 0
		for i in range(self.model1.dim_latent):
			loss += (float(y_pred1[i]) - float(y_pred2[i]))**2

		return loss


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

		torch.save(self.model1.cpu(), self.outPath + "model1.data")
		torch.save(self.model2.cpu(), self.outPath + "model2.data")

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

	def is_over_fitting(self):
		"""Returns True is the modele is overfitting.
		The model is considered overfitting if the loss in respect to the test data is 
		not decreasing for T epoch, with a threshold of K.
		
		Returns
		-------
		bool
			True if the model is overfitting, False otherwise.
		"""

		return False

	def myloss(self, batch, alpha=0.7):

		# Loss we use for the training and the evaluation, same that the one used in the paper
		# pairwise hinge loss, using cosine similarity

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

		# Construct a dictionary allowing to match coordonate in the latent space with a 
		# name of music piece and another one that match names with pianoroll matrix.


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
		# Learn mathod, that train the model on self.batches

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

			if self.losses_test[t] < self.lastloss:
				self.save_weights()
				self.lastloss = self.losses_test[t]

			if self.is_over_fitting():
				# stop learning
				return



		self.plot_losses()

		self.model1 = torch.load(self.outPath + "model1.data")
		self.model2 = torch.load(self.outPath +  "model2.data")

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

		# Construct a dictionary allowing to match coordonate in the latent space with a 
		# name of music piece and another one that match names with pianoroll matrix.
		# This function only do this for the validation set of data

		print("____ Consctructing the dictionary")

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
		# Search for the nearest neighbor in the evaluation set

		dist = 1000000
		name = ""
		for key in self.dicoEval:
			tmp_dist = self.s(key, wavePosition[0])
			if  tmp_dist < dist:
				dist = tmp_dist
				name = self.dicoEval[key]

		return name


	def nearestNeighbor(self, wavePosition):

		# Search for the nearest neighbor in the training set

		dist = 1000000
		name = ""
		for key in self.dico:
			tmp_dist = self.s(key, wavePosition[0])
			if  tmp_dist < dist:
				dist = tmp_dist
				name = self.dico[key]

		return name

	def pianorollDistance(self, p1, p2):
		# Compute a distance between two pianoroll, using pairwise comparison only on non-zero elements
		# in order to avoid false high similarity due to spasity.

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
		# Process an evaluation test we designed. It return the meaned quantity of non-zeros pianoroll element 
		#that have been correctly predicted. Only on the evaluation set.

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
	# Process an evaluation test we designed. It return the meaned quantity of non-zeros pianoroll element 
	#that have been correctly predicted. Only on the training set.

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
		# compute the recall R@k for one element.

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
		# Return the meaned recall R@k for the all validation set

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
		# Compute the MRR (cf. article) for one element.

		recall = {}
		for key in self.dicoEval:
			recall[self.s(key, embedded[0])] = self.dicoEval[key]

		keylist = list(recall.keys())
		keylist.sort()

		for i in range(len(keylist)):
			if np.array_equal(self.rollsFromNameEval[recall[keylist[i]]], expected):
				return 1/(i+1)

	def getMRR(self):
		# Compute the mRR for all elemtns from validation set.
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
		# Compute the MR for one element.

		recall = {}
		for key in self.dicoEval:
			recall[self.s(key, embedded[0])] = self.dicoEval[key]

		keylist = list(recall.keys())
		keylist.sort()

		for i in range(len(keylist)):
			if np.array_equal(self.rollsFromNameEval[recall[keylist[i]]], expected):
				return i+1

	def getMR(self):
		# Compute the MR for all element on validation set.

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
		# Return the name of predicted song.

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
		# Add all pianoroll snippets extracted from a given file

		windowSize = 4
		STEP = 2
		snippets = []

		self.model1.eval()

		s = score.score(file)
		if computeSound is True:
			file = file[file.rfind("/")+1:]
			file = file[:file.rfind(".")]
			s.toWaveForm().save(pathTemp +  file + ".wav")

		windowSize *= s.quantization
		N = s.length * s.quantization

		for i in range((N-windowSize)//STEP):
			tmpPart1 = s.extractPart(i*STEP, i*STEP+windowSize)
			N1 = np.array(tmpPart1.getPianoRoll()).astype(float)
			N1 = N1.reshape(1, 1, N1.shape[0], N1.shape[1])
			X1 = torch.FloatTensor(N1)  
			embedded = self.model1.forward(X1)[0]

			self.dico[embedded] = tmpPart1.name


