import sys 
sys.path.append('../')

from Modules import dataBase

## Construct and save database
D1 = dataBase.dataBase()
D1.constructDatabase("dataBaseTest") # on a tiny dataset
#D1.constructDatabase("../DataBase") # on the whole database

#D1.print()
print(len(D1.getData()))
print()
print("End of dataset Construction")
print()

quit()
D = D1.getBatches(64)

for elem in D:
	print(elem[3])

quit()

D = D1.getTestSet(32)
for elem in D:
	print(elem[2])

D = D1.getValidationSet(32)
for elem in D:
	print(elem[2])	

D1.save(path="dataBaseTest/Serialized/")
D1.print()

# load a previous saved database
D2 = dataBase.dataBase()
D2.load("dataBaseTest/Serialized/dataBaseTest.data")
D2.print()
