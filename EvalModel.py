from Model import modele

import torch
import datetime
import sys
import pickle
import os
from glob import glob
from optparse import OptionParser
import warnings

warnings.filterwarnings("ignore")

def evalSongDetection(model, folder, pathTemp="/fast/guilhem/"):

	score = 0
	N = 0

	for file in glob(folder+'**', recursive=True):
		if file[file.rfind("."):] in [".mid", ".midi"] and os.path.isfile(file):
			file = file[file.rfind("/")+1:]
			file = file[:file.rfind(".")]

			name = model.whatIsThisSong(pathTemp + file + ".wav")
			name = name[:name.rfind("_")]
			name = name[:name.rfind("_")]

			if name == file:
				score += 1
			N += 1

	print("We made good detection of", score, "files on", N, ".")


def addToDictionary(model, folder, pathTemp="/fast/guilhem/"):
	for file in glob(folder+'**', recursive=True):
		if file[file.rfind("."):] in [".mid", ".midi"] and os.path.isfile(file):
			model.addToDictionary(file, pathTemp=pathTemp)

def evalOnTest(database, modelsPath, outPath="/fast-1/guilhem/", name="", gpu=None, testFolder="DataBaseForTest/"):

	if name == "":
		now = datetime.datetime.now()
		data = "eval_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + ".data"

	model = modele.Modele(database, gpu=gpu, outPath=outPath)

	model.model1 = torch.load(modelsPath + "model1.data")
	model.model2 = torch.load(modelsPath + "model2.data")

	if model.GPU:
		model.model1 = model.model1.cuda()
		model.model2 = model.model2.cuda()

	addToDictionary(model, testFolder, pathTemp=outPath)

	print()
	print("Test Loss for the best trained model:", model.TestEval(model.testBatches))

	print()
	model.constructDictForTest()
	model.constructDict()


	print("___ Benchmark")

	print("Recall1:", model.getRecallK(1))
	print("Recall25:", model.getRecallK(25))
	print("MRR:", model.getMRR())
	print("MR:", model.getMR())
	print("Reconstruction score:", model.testBenchmarkEval())
	print("Reconstruction score on unknwon snipets:", model.testBenchmark())
	evalSongDetection(model, testFolder, pathTemp=outPath)




	#pickle.dump((model.TestEval(model.testBatches), score), open(name, "wb" ) )


if __name__ == "__main__":

	usage = "usage: %prog [options] <path to database> <folder for model>"
	parser = OptionParser(usage)

	parser.add_option("-t", "--testFolder", type="string",
	                  help="Path for the folder containing the songs to detect, by default it's DataBaseForTest/", 
	                  dest="testFolder", default="DataBaseForTest/")

	parser.add_option("-o", "--outPath", type="string",
	                  help="Path for the temporary folder.", 
	                  dest="outPath", default="/fast-1/guilhem/")

	parser.add_option("-n", "--name", type="string",
	                  help="Name for the output file", 
	                  dest="name", default="")

	parser.add_option("-g", "--gpu", type="int",
	                  help="ID of the GPU, run in CPU by default.", 
	                  dest="gpu")



	options, arguments = parser.parse_args()
	
	if len(arguments) == 2:
		evalOnTest(arguments[0], arguments[1], outPath=options.outPath, name=options.name, gpu=options.gpu, testFolder=options.testFolder)

	else:
		parser.error("You have to specify the path of the database and the models.")


