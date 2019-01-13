import torch
import numpy as np

def loss(batch, alpha=0.7):

	s = torch.nn.CosineSimilarity(dim=0)
	X1, X2, L1, L2, indices = batch

	rank = 0

	for x in range(len(X1)):
		for y in range(len(X2)):
			if y != x:
				rank += max(0, alpha - s(X1[indices[x]], X2[x]) + s(X1[indices[x]], X2[y]))

	return rank


X1 = torch.randn(32, 32) 
X2 = torch.randn(32, 32) 
L1 = []
L2 = []
indices = np.arange(32)
np.random.shuffle(indices)
X2 = X2[indices]

batch = (X1, X2, L1, L2, indices)

print("___ Should be high")

print(loss(batch))

X1 = torch.randn(32, 32) 
X2 = torch.randn(32, 32) 
for i in range(len(X2)):
	X2[i] = X1[i]
L1 = []
L2 = []
indices = np.arange(32)
np.random.shuffle(indices)
X2 = X2[indices]

batch = (X1, X2, L1, L2, indices)

print("___ Should be low")
print(loss(batch))

