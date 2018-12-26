import os

class dataBase:
	def __init__(self, path):

		if os.path.isdir(path):
			self.path = path
			print("We are working on '" + path + "'")
		else:
			print("The path you gave is not a directory, please provide a correct directory.")
			raise RuntimeError("Invalid database directory")




dataBase("database")