from Model import modele

import torch
import sys
from optparse import OptionParser

def main(database, EPOCHS=20, gpu=None, outPath=".TEMP/", learning_rate=1e-2):

	model = modele.Modele(database, gpu=None, outPath=".TEMP/")

	model.learn(EPOCHS, learning_rate=1e-2)


if __name__ == "__main__":

	usage = "usage: %prog [options] <path to database>"
	parser = OptionParser(usage)

	parser.add_option("-e", "--epochs", type="int",
	                  help="Number of Epochs",
	                  dest="epochs", default=20)

	parser.add_option("-g", "--gpu", type="int",
	                  help="ID of the GPU, run in CPU by default.", 
	                  dest="gpu")

	parser.add_option("-o", "--outPath", type="string",
	                  help="Path for the temporary folder.", 
	                  dest="outPath", default=".TEMP/")

	parser.add_option("-l", "--learning_rate", type="float",
	                  help="Value of the starting learning rate", 
	                  dest="learning_rate", default=1e-2)

	options, arguments = parser.parse_args()
	
	if len(arguments) == 1:
		main(arguments[0], EPOCHS=options.epochs, gpu=options.gpu, outPath=options.outPath, learning_rate=options.learning_rate)

	else:
		parser.error("You have to specify the path of the database.")


