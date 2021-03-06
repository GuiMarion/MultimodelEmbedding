from Modules import waveForm
from Model import modele
from Modules import score

import numpy as np
import torch
import pickle
import sys
from tqdm import tqdm


def nearestNeighbor(dico, wavePosition):

	s = torch.nn.CosineSimilarity(dim=0)
	dist = 1000000
	name = ""
	for key in dico:
		tmp_dist = 1 - s(key.cpu(), wavePosition[0])
		if  tmp_dist < dist:
			dist = tmp_dist
			name = dico[key]

	return name

def getMidiFromWave(folder, file, dataBase="DataBase/MIDIs/mozart/"):
	
	window_size = 43

	model2 = torch.load(folder + "model2.data")
	model2.eval()

	dico = pickle.load(open(folder + "dico.data", "rb"))[0]

	w = waveForm.waveForm(file).getCQT()

	CQTs = []
	for i in range(len(w[0])//172 -1):
		CQTs.append(w[ : , i*window_size : (i+1)*window_size])
		CQTs[i] = np.array(CQTs[i]).astype(float)
		CQTs[i] = CQTs[i].reshape(1, 1, CQTs[i].shape[0], CQTs[i].shape[1])
		CQTs[i] = torch.FloatTensor(CQTs[i])

	wraps = []
	for i in tqdm(range(len(CQTs[:10]))):
		wraps.append(nearestNeighbor(dico, model2.forward(CQTs[i]).data))

	pianoroll = None

	for elem in wraps:
		end = int(elem[elem.rfind("_")+1:])
		name = elem[:elem.rfind("_")]
		start = int(name[name.rfind("_")+1:])
		name = name[:name.rfind("_")]	

		s = score.score(dataBase + name + ".mid")
		s = s.extractPart(start, end)

		if pianoroll is None:
			pianoroll = s.pianoroll
		else:
			pianoroll = np.concatenate((pianoroll, s.pianoroll), axis=0)

	out = score.score("", fromArray=(pianoroll, "out"))

	out_name = file[:file.rfind(".")] + ".mid"
	out.writeToMidi(out_name)

	print("File saved in", out_name)


if __name__ == "__main__":

	if len(sys.argv) == 3 :
		getMidiFromWave(sys.argv[1], sys.argv[2])
	elif len(sys.argv) == 4:
		getMidiFromWave(sys.argv[1], sys.argv[2], dataBase=sys.argv[3])

	else:
		print("Usage: Python3 WaveToMidi.py <folder for model> <file to convert>")  





