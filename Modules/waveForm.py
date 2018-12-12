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
			self.loadFromFile(path) # retrieves data from path
			self.length = len(self.data) / self.sampleRate # data's duration in seconds
			self.tempo = 0 # Comment récupérer le tempo?


		else:
			self.tempo = 0
			self.sampleRate = 0
			self.data = []
			self.length = 0

		self.name = os.path.splitext(os.path.basename(path))[0] # reads the file's name
		
	def loadFromFile(self, path): #loads data from a file with a specified path
		# update the attributes, should use loadFrom data
		self.data, self.sampleRate = sf.read(path)

		return "Fichier chargé."


	def loadFromData(self, sequence, sampleRate): #loads data from a sequence with a specifided sampleRate
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

		sf.write(path, self.data, self.sampleRate)

		return "Fichier sauvegardé."

	def plot(self):
		# plot the signal
		plt.figure()

		t = np.linspace(0, self.length, len(self.data) ) # Time vector

		plt.figure(1)

		plt.plot(t, self.data)
		plt.xlabel('Time')
		plt.ylabel('Amplitude')
		plt.title('Wave Form')

		plt.show()


	def getFFT(self):
		# return FFT if already computed or compute it, store it and return it
		# store the result of the FFT as attribute in order to compute it only once

		self.FFT = np.fft.fft(self.data)
		#self.freq = np.linspace(0, self.sampleRate / 2, len(self.data) / 2) # Freq vector
		self.freq = np.fft.fftfreq(len(self.data))



	def plotFFT(self):
		# plot FFT

		plt.figure(2)

		plt.plot(self.freq, np.abs(self.FFT))
		plt.xlabel('Frequencies')
		plt.ylabel('Amplitude')
		plt.title('FFT')

		plt.show()



