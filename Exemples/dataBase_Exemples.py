import sys 
sys.path.append('../')

from Modules import dataBase

## Construct and save database
D1 = dataBase.dataBase()
D1.constructDatabase("dataBaseTest") # only on the short database
#D1.constructDatabase("../DataBase") # on the whole database
D1.save()
D1.print()

# load a previous saved database
D2 = dataBase.dataBase()
D2.load("../DataBase/Serialized/dataBaseTest.data")
D2.print()


dico = D2.get()
for key in dico:
	# should plot the scores
	dico[key][0].plot()
	