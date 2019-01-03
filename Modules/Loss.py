import numpy as np
import torch
from Modules import dataBase

# We first define the cosine similarity
def cos_sim(a,b):
	
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	
	return dot_product / (norm_a * norm_b)

# Then we compute the loss which is a function of both weight vectors w_audio and w_score
# We find the right weights by finding the min of the loss 
def L_rank(database, w_audio, w_score, alpha) : 

	dico = database.get()
	
	# x and y are N samples * 32 tensors
	# w_audio is a 1*32 vector with the weights of the audio net
	# w_score is a 1*32 vector with the weights of the score net
	
	N = len(dico)	# number of samples
	L = 0			# initialize the loss

	for key in dico:
		for mismatching in dico : 
			
			if mismatching != key : 	
				x = dico[key][0].getPianoRoll()					# get the pianoroll of the score sample 
				y = dico[key][1].getFFT()						# get the FFT (TFCT) of the audio sample
				y_mismatching = dico[mismatching][1].getFFT()	# get the FFT (TFCT) of the mismatching audio samples

				# Element wise multiplication of the samples with the right weights
				x = torch.mul(x, w_audio)
				y = torch.mul(y, w_score)
				y_mismatching = torch.mul(y_mismatching, w_score)
	

				L += max(0,alpha - cos_sim(x,y) + cos_sim(x,y_mismatching))
			
	return L