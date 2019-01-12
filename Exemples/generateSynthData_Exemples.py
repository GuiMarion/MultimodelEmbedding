import sys 
sys.path.append('../')
import numpy as np

from Modules import generateSyntheticData as gen


# this script help you to understand the usages for the generateSyntheticData one. 
# It generate random data but with the exact same structure than our reel data

# You can generate data with a default shape for the pianoroll and tfct
# data = gen.generateSyntData(10, 2)

# for i in range(len(data)):
# 	print("Pianoroll: ")
# 	print(data[i][0])
# 	print("TFCT:")
# 	print(data[i][1])
# 	print("Name:")
# 	print(data[i][2])
# 	print()


# # And you call also specify the shapes by an extra parameter
# data = gen.generateSyntData(10,2, size1=(1, 1), size2=(1,2))

# for i in range(len(data)):
# 	print("Pianoroll: ")
# 	print(data[i][0])
# 	print("TFCT:")
# 	print(data[i][1])
# 	print("Name:")
# 	print(data[i][2])
# 	print()

for i in range(1000):
	N = np.random.randint(low= 32, high=1000)
	r = gen.testMyBatchFunction(N, 32)
	if r == False:
		print("NIKE TA MERE CA MARCHE PAS !!!!!!!!")
		break