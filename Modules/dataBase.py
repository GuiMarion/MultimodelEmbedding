import os
from glob import glob
import pickle

from Modules import score
from Modules import waveForm

class dataBase:
	def __init__(self, name=None):

		self.name = name
		self.dico = None
		self.path = None


	def constructDatabase(self, path, name=None):
		# construct a dictionary wich contains, for every file, a tuple with the corresponding score and waveForm object

		if os.path.isdir(path):
			self.path = path
			if name:
				self.name = name
			else:
				self.name = str(path)
			print()
			print("________ We are working on '" + path + "'")
			print()
		else:
			print("The path you gave is not a directory, please provide a correct directory.")
			raise RuntimeError("Invalid database directory")

		print("_____ Filling the database ...")
		print()

		# Number of skiped files
		skipedFiles = 0
		# Total number of files
		N = 0
		dico = {}
		for filename in glob(self.path+'/**', recursive=True):

			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				if os.path.isfile(filename):
					print("	-", filename)
					try : 
						score_temp = score.score(filename)
						dico[filename] = (score_temp, score_temp.toWaveForm())
					except RuntimeError:
						skipedFiles += 1
					N += 1

		self.dico = dico

		print()
		print("We passed a total of ", N, "files.")
		print(skipedFiles,"of them have been skiped.")
		print()


	def save(self, path="../DataBase/Serialized/"):
		# Save database as a pickle

		answer = "y"

		if os.path.isfile(path+self.name+'.data'):
			answer = str(input("'"+path+self.name+'.data'+"'" + " already exists, do you want to replace it ? (Y/n)"))

			while answer not in ["", "y", "n"]:
				answer = str(input("We didn't understand, please tape enter, 'y' or 'n'"))

		if answer in ["", "y"]:
			pickle.dump(self.dico, open(path+self.name+'.data','wb'))
			print()
			print("The new database is saved.")
		else:
			print()
			print("We kept the old file.")


	def load(self, path):
		# Load a database from a previous saved pickle
		if not os.path.isfile(path):
			print("The path you entered doesn't point to a file ...")
			raise RuntimeError("Invalid file path")

		try:
			self.dico = pickle.load(open(path, 'rb'))
			print("We sucessfully loaded the database.")
			print()
		except (RuntimeError, UnicodeDecodeError) as error:
			print("The file you provided is not valid ...")
			raise RuntimeError("Invalid file")

	def print(self):
		# Print name of all items in database
		print("____Printing database")
		print()
		for key in self.dico:
			print("	-", self.dico[key][0].name)

	def get(self):
		return self.dico

	def augmentWithTranspo(self):
		# tranpose every piece of the dico in order to extend it
		# len(dico) should be 12 times bigger after the function then before the functions

		print("TODO")

