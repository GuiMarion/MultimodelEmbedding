import sys 
sys.path.append('../')

from Modules import generateSyntheticData as gen


# this script help you to understand the usages for the generateSyntheticData one. 
# It generate random data but with the exact same structure than our reel data

# You can generate data with a default shape for the pianoroll and tfct
data = gen.generateSyntData(10, 2)

for i in range(len(data)):
	print("Pianoroll: ")
	print(data[i][0])
	print("TFCT:")
	print(data[i][1])
	print("Name:")
	print(data[i][2])
	print()


# And you call also specify the shapes by an extra parameter
data = gen.generateSyntData(10,2, size1=(1, 1), size2=(1,2))

for i in range(len(data)):
	print("Pianoroll: ")
	print(data[i][0])
	print("TFCT:")
	print(data[i][1])
	print("Name:")
	print(data[i][2])
	print()

gen.testMyBatchFunction(data, 32)