from Model import modele

import torch
import datetime
import sys
import pickle

def evalOnTest(database, modelsPath, name="", gpu=None):

	if name == "":
		now = datetime.datetime.now()
		data = "eval_" + now.month + "_" + now.day + "_" + now.hour+".data"

	model = modele.Modele(database, gpu=gpu, outPath="/fast-1/guilhem/")

	model.model1 = torch.load(modelsPath + "model1.data")
	model.model2 = torch.load(modelsPath + "model2.data")

	if model.GPU:
		model.model1 = model.model1.cuda()
		model.model2 = model.model2.cuda()

	print()
	print("Test Loss for the best trained model:", model.TestEval(model.testBatches))

	print()
	model.constructDict()

	print()
	score = model.testBenchmark()

	print("Benchmark score:", score)
	print()

	pickle.dump((model.TestEval(model.testBatches), score), open(name, "wb" ) )


if __name__ == "__main__":

	if len(sys.argv) == 3 :
		evalOnTest(sys.argv[1], sys.argv[2])
	elif len(sys.argv) == 4:
		evalOnTest(sys.argv[1], sys.argv[2], name=sys.argv[3])
	elif len(sys.argv) == 5:
		evalOnTest(sys.argv[1], sys.argv[2], name=sys.argv[3], gpu=int(sys.argv[4]))

	else:
		print("Usage: Python3 EvalModel.py <database> <folder for model> <name -optional> <gpu -optional>")  



