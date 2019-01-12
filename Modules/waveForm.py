import sounddevice as sd # On importe sounddevice
import os
import soundfile as sf
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack, coo_matrix
import librosa

class waveForm:
	def __init__(self, path):

		if path is not None:
			self.data = None
			self.sampleRate = None
			self.loadFromFile(path) # retrieves data from path
			self.length = len(self.data) / self.sampleRate # data's duration in seconds
			self.tempo = 0 # Comment récupérer le tempo?
			self.FFT = None
			self.STFT = None
			self.STFTlog = None
			self.CQT = None

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

	def getData(self):
		return self.data

	def play(self, length=None):
		# play the data, use the module sound device
		if length is None:
			length = len(self.data)//self.sampleRate
		sd.play(self.data, self.sampleRate)
		time.sleep(length)

	def save(self, path):
		# save as a wav file at path

		sf.write(path, self.data, self.sampleRate)

		return "Fichier sauvegardé."

	def plot(self):
		# plot the signal
		t = np.linspace(0, self.length, len(self.data) ) # Time vector

		plt.figure()

		plt.plot(t, self.data)
		plt.xlabel('Time')
		plt.ylabel('Amplitude')
		plt.title('Wave Form')

		plt.show()


	def getFFT(self):
		# return FFT if already computed or compute it, store it and return it
		# store the result of the FFT as attribute in order to compute it only once

		self.FFT = np.fft.fft(self.data)
		self.freq = np.fft.fftfreq(len(self.data))


	def plotFFT(self):
		# plot FFT

		if self.FFT is None:
			self.getFFT()

		plt.figure()

		plt.plot(self.freq, np.abs(self.FFT))
		plt.xlabel('Frequencies')
		plt.ylabel('Amplitude')
		plt.title('FFT')

		plt.show()

	def getSTFT(self):
		if self.STFT is None:
			self.computeSTFT()
		return self.STFT, self.STFTsec, self.STFTfreq

	def computeSTFT(self, L_n = 4096):
		STEP_n = int(self.sampleRate // 20)
		Nfft =  L_n * 4
		nLim = int((len(self.data)-L_n) / (STEP_n))

		Fft_m = np.zeros((Nfft, nLim))
		self.STFT = np.zeros((round(Nfft/2)+1, nLim))
		self.STFTfreq = np.linspace(1, self.sampleRate/2, round(Nfft/2)+1)
		self.STFTsec = np.linspace(0, self.length, nLim)

		window = np.hamming(L_n)
		self.data = self.data[:,0]

		for fen in range(nLim):
			Fft_m[:,fen] = np.fft.fft(window * self.data[fen*STEP_n:fen*STEP_n+L_n], Nfft)
			self.STFT[:,fen] = np.abs(Fft_m[0:round(Nfft/2)+1, fen])

	def plotSTFT(self):
		if self.STFT is None:
			self.computeSTFT()
		plt.figure()
		#plt.imshow(np.sqrt(self.STFT), origin='lower', aspect='auto', extent=[self.STFTsec[0], self.STFTsec[-1], self.STFTfreq[0], self.STFTfreq[-1]], interpolation='nearest')
		plt.pcolormesh(self.STFTsec, self.STFTfreq, np.sqrt(self.STFT))
		plt.ylim((30, 6000))
		plt.xlabel('Time(s)')
		plt.ylabel('Fréquence(Hz)')
		plt.show()

	def getSTFTlog(self):
		if self.STFTlog is None:
			self.computeSTFTlog()
		return self.STFTlog

	def computeSTFTlog(self):
		if self.STFT is None:
			self.computeSTFT()

		f0 = 30
		dF = self.sampleRate / len(self.STFTfreq)
		self.STFTlog = np.zeros((128, len(self.STFTsec)))
		for bin in range(0,128):
			freq_c = np.power(2, (bin + 1)/16) * f0
			down_lim = int(np.floor(freq_c * np.power(2, -1/32)) / dF)
			up_lim = int(np.ceil(freq_c * np.power(2, 1/32)) / dF) + 1
			for fen in range(0, len(self.STFTsec)-1):
				tronc = np.array(self.STFT[down_lim : up_lim, fen])
				self.STFTlog[bin,fen] = np.mean(tronc) / (up_lim - down_lim)


	def computeCQT(self):
		vect = self.data[:,0]
		self.CQT = np.abs(librosa.cqt(vect, sr=self.sampleRate, fmin=30, n_bins=128, bins_per_octave=16))

	def getCQT(self):
		if self.CQT is None:
			self.computeCQT()
		return self.CQT

	def plotCQT(self):
		if self.CQT is None:
			self.computeCQT()

		plt.figure()
		plt.pcolormesh(np.arange(0,len(C[0,:])), np.arange(0,len(C[:,0])), C)
		plt.colorbar(format='%+2.0f dB')
		plt.tight_layout()
		plt.xlabel('Time')
		plt.ylabel('Frequency bin')
		plt.show()
