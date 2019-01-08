import sys 
sys.path.append('../')

from Modules import dataBase

## Construct and save database
D1 = dataBase.dataBase()
D1.constructDatabase("dataBaseTest") # on a tiny dataset
#D1.constructDatabase("../DataBase") # on the whole database
D1.save("dataBaseTest/Serialized/")
D1.print()

# load a previous saved database
D2 = dataBase.dataBase()
D2.load("dataBaseTest/Serialized/dataBaseTest.data")
D2.print()

dico = D2.getData()
print("\nPlaying and displaying snippets :")
print()

for key in dico:
	print(key[0].name + '♪')
	# should plot the scores
	for minibout in key[1]:
		minibout.play()
	key[0].plot()
	
