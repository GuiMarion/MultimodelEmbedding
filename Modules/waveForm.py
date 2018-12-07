import sounddevice as sd # On importe sounddevice
import os
import soundfile as sf
import time
import numpy as np
import matplotlib.pyplot as plt

class waveForm:
	def __init__(self, path):

		if path is not None:
			self.data = None
			self.sampleRate = None
			self.loadFromFile(path)
			self.length = len(self.data) / self.sampleRate


		else:
			self.tempo = 0
			self.sampleRate = 0
			self.data = []
			self.length = 0

		self.name = os.path.splitext(os.path.basename(path))[0]
		
	def loadFromFile(self, path):
		# update the attributes, should use loadFrom data
		self.data, self.sampleRate = sf.read(path)

		return "Fichier chargé."


	def loadFromData(self, sequence, sampleRate):
		# update the attributes
		# input sequence is a vector of samples
		# should work with mono as well as with stereo 

		self.data = sequence
		self.sampleRate = sampleRate
		self.length = len(sequence) / sampleRate

		return "Données chargées."


	def play(self, length=10):
		# play the data, use the module sound device
		
		sd.play(self.data, self.sampleRate)
		time.sleep(length)

	def save(self, path):
		# save as a wav file at path

		return "Fichier sauvegardé."

	def plot(self):
		# plot the signal
		plt.figure()
		t = np.linspace(0, self.length, len(self.data) )
		plt.plot(t, self.data)
		plt.show()


	def getFFT(self):
		# return FFT if already computed or compute it, store it and return it
		# store the result of the FFT as attribute in order to compute it only once

		self.FFT = np.fft.fft(self.data, nextpow2(len(self.data)))
		self.freq = np.linspace(0, self.sampleRate / 2, nextpow2(len(self.data)) / 2)

		return "TODO"

	def plotFFT(self):
		# plot FFT

		return "TODO"