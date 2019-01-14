from Modules import dataBase

import numpy as numpy
import torch

def getDistancePianoroll(p1, p2):

	for i in range(len(p1)):
		for j in range(len(p1[0])):
			if p1[i][j] != p2[i][j]



print("____ Loading batches from file")

## Construct and save database
D = dataBase.dataBase()
D.constructDatabase(self.databasePath)
self.batches = D.getBatches(self.batch_size)

self.testBatches = D.getTestSet(self.batch_size)

self.validationBatches = D.getValidationSet(self.batch_size)

model1 = torch.load("/fast-1/guilhem/params/model1.data")
model2 = torch.load("/fast-1/guilhem/params/model2.data")

model1.eval()
model2.eval()

