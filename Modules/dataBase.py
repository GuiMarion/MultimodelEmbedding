import os
from glob import glob
import pickle
from tqdm import tqdm

from Modules import score
from Modules import waveForm

# Parameters for the data extraction part
WINDOW_SIZE = 5 # in beat
STEP = 120 # in sample

FONTS = ["000_Florestan_Piano.sf2"] # TODO add more fonts

class dataBase:
	def __init__(self, name="database"):

		self.name = name
		self.data = []
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
		scores = []
		for filename in glob(self.path+'/**', recursive=True):

			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				if os.path.isfile(filename):
					print("	-", filename)
					try : 
						score_temp = score.score(filename)
						scores.extend(score_temp.extractAllParts(WINDOW_SIZE, step=STEP))
						
					except RuntimeError:
						skipedFiles += 1
					N += 1

		print()
		print("We passed a total of ", N, "files.")
		print(skipedFiles,"of them have been skiped.")
		print()

		print("_____ Augmenting database ...")
		print()

		#scores = self.augmentData(scores)

		print("_____ Computing the sound ...")
		print()

		for s in tqdm(scores):
			waveforms = []
			for font in FONTS:
				waveforms.append(s.toWaveForm(font=font))
			self.data.append((s, waveforms))


	def save(self, path="../DataBase/Serialized/"):
		# Save database as a pickle

		answer = "y"

		if os.path.isfile(path+self.name+'.data'):
			answer = str(input("'"+path+self.name+'.data'+"'" + " already exists, do you want to replace it ? (Y/n)"))

			while answer not in ["", "y", "n"]:
				answer = str(input("We didn't understand, please type enter, 'y' or 'n'"))

		if answer in ["", "y"]:
			print("____ Saving database ...")
			pickle.dump(self.data, open(path+self.name+'.data','wb'))
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
			self.data = pickle.load(open(path, 'rb'))
			print("We successfully loaded the database.")
			print()
		except (RuntimeError, UnicodeDecodeError) as error:
			print("The file you provided is not valid ...")
			raise RuntimeError("Invalid file")

	def print(self):
		# Print name of all items in database
		print("____Printing database")
		print()
		for i in range(len(self.data)):
			print(self.data[i][0].name)

	def getData(self):
		return self.data

	def augmentData(self, scores):
		# augment the data with some techniques like transposition
		
		augmentedData = []

		for s in scores:
			augmentedData.extend(s.getTransposed())


		return augmentedData


