import os
import soundfile as sf
import time
import numpy as np
import librosa

try:
	import matplotlib.pyplot as plt
	plot = True
except ImportError:
	plot = False

try:
	import sounddevice as sd
	sound = True
except ImportError:
	sound = False


class waveForm:
	"""This class is used to manage waveform audio data from the dataset.

	Each class is associated with a unique wave file. Then, a dimensionnaly
	reduced spectrogram or CQT can be computed.

	Attributes
	----------
	self.data : :obj:'list' of :obj:'float'
	    Contains the sound data itselfs in an array.
	self.sampleRate : int
		Sample Rate of the data.
	self.length : int
		Length of the data in seconds.
	self.FFT : :obj:'list' of :obj:'float'
		Complex Fourier transform of the data.
	self.STFT : :obj:'list' of :obj:'float'
		Power-spectrogram of the data.
	self.STFTlog : :obj:'list' of :obj:'float'
		A dimensionnaly reduced spectrogram based on a logaritmic band-filter of
		self.STFT
	self.CQT : :obj:'list' of :obj:'float'
		Constant Quality factor Transform of the data. A good alternative to the
		STFT.
	self.STFTsec : :obj:'list' of :obj:'float'
		Computed with computeSTFT. Vector containing temporal indicies of STFT.
	self.STFTfreq : :obj:'list' of :obj:'float'
		Computed with computeSTFT. Vector containing frequencial indicies of
		STFT.
	"""

	def __init__(self, path):
		""" Initialises the waveform class.

		Note that only self.data, self.sampleRate and self.length are computed
		from itself.

		Parameters
		----------
		path : str
			Path of the audio data associated to the class.
		"""

		if path is not None:
			self.data = None
			self.sampleRate = None
			self.loadFromFile(path) # retrieves data from path
			self.length = len(self.data) / self.sampleRate
			self.FFT = None
			self.STFT = None
			self.STFTlog = None
			self.CQT = None

		else:
			self.tempo = 0
			self.sampleRate = 0
			self.data = []
			self.length = 0

		self.name = os.path.splitext(os.path.basename(path))[0]

	def loadFromFile(self, path):
		""" Loads data from a file. Update the attributes to initializationself.

		Only loadFromData should be used.

		Parameters
		----------
		path : str
			Path of the audio data associated to the class.
		"""
		self.data, self.sampleRate = sf.read(path)


	def loadFromData(self, sequence, sampleRate):
		""" Loads data from a specified vector, assigns it to the class.

		Should work for a mono file as well as a stereo one.

		Parameters
		----------
		sequence : :obj:'list' of :obj:'float'
			Input vector.
		sampleRate : int
			Sample Rate of the input vector.
		"""

		self.data = sequence
		self.sampleRate = sampleRate
		self.length = len(sequence) / sampleRate

	def getData(self):
		""" Returns raw data."""
		return self.data

	def play(self, length=None):
		""" Plays the data.

		Parameters
		----------
		length : float
			Duration of the data to play. All of it by default.
		"""
		if sound is True:
			if length is None:
				length = len(self.data)//self.sampleRate
			sd.play(self.data, self.sampleRate)
			time.sleep(length)
		else:
			print("We cannot play any sound on this device.")

	def save(self, path):
		""" Save in a precised path the data into a wave file.

		Parameters
		----------
		path : str
			Were to save the data.
		"""

		sf.write(path, self.data, self.sampleRate)

	def plot(self):
		""" Plots the signal."""

		t = np.linspace(0, self.length, len(self.data) ) # Time vector

		plt.figure()
		plt.plot(t, self.data)
		plt.xlabel('Time')
		plt.ylabel('Amplitude')
		plt.title('Wave Form')

		plt.show()


	def getFFT(self):
		""" Return and store the complex Fourier transform from data."""

		self.FFT = np.fft.fft(self.data)
		self.freq = np.fft.fftfreq(len(self.data))


	def plotFFT(self):
		""" Plot Fourier transform from self.FFT. Compute it if necessary."""

		if self.FFT is None:
			self.getFFT()

		plt.figure()
		plt.plot(self.freq, np.abs(self.FFT))
		plt.xlabel('Frequencies')
		plt.ylabel('Amplitude')
		plt.title('FFT')

		plt.show()

	def getSTFT(self, L_n = 4096):
		""" Returns the spectrogram of the data, computed with computeSTFT.

		By default, window's size of the several FFT is 4096, with a zero
		padding up to four times the window's size. All the Fourier transforms
		are computed with a Hamming window, and the step factor is approximately
		20 per second.

		Parameters
		----------
		L_n : int
			Window's size of the FFTs. 4096 by default.
		"""

		if self.STFT is None:
			self.computeSTFT(L_n)
		return self.STFT, self.STFTsec, self.STFTfreq

	def computeSTFT(self, L_n = 4096):
		STEP_n = int(self.sampleRate // 20)
		Nfft =  L_n * 4
		nLim = int((len(self.data)-L_n) / (STEP_n))

		Fft_m = np.zeros((Nfft, nLim), dtype='complex64')
		self.STFT = np.zeros((round(Nfft/2)+1, nLim))
		self.STFTfreq = np.linspace(1, self.sampleRate/2, round(Nfft/2)+1)
		self.STFTsec = np.linspace(0, self.length, nLim)

		window = np.hamming(L_n)
		self.data = self.data[:,0]

		for fen in range(nLim):
			Fft_m[:,fen] = np.fft.fft(window * self.data[fen*STEP_n:fen*STEP_n+L_n], Nfft)
			self.STFT[:,fen] = np.abs(Fft_m[0:round(Nfft/2)+1, fen])

	def getSTFTlog(self, b=16):
		""" Return a dimensionnaly reduced spectrogram.

		Converts a classic STFT (self.STFT) by operating logaritmic filter
		bands and, for each temporal sample of the TFCT, computing the mean in
		the resulting frequency bin. Computed with computeSTFTlog.

		Parameters
		----------
		b : int
			Number of bands per octave used to the reduction of the STFT. 16 by
			default. Influences the frequency resolution.
		"""
		if self.STFTlog is None:
			self.computeSTFTlog(b)
		return self.STFTlog

	def computeSTFTlog(self, b=16):
		if self.STFT is None:
			self.computeSTFT()

		f0 = 30
		dF = self.sampleRate / len(self.STFTfreq)
		self.STFTlog = np.zeros((128, len(self.STFTsec)))
		for bin in range(0,128):
			freq_c = np.power(2, (bin + 1)/b) * f0
			down_lim = int(np.floor(freq_c * np.power(2, -1/(2*b))) / dF)
			up_lim = int(np.ceil(freq_c * np.power(2, 1/(2*b))) / dF) + 1
			for fen in range(0, len(self.STFTsec)-1):
				tronc = np.array(self.STFT[down_lim : up_lim, fen])
				self.STFTlog[bin,fen] = np.mean(tronc) / (up_lim - down_lim)

	def plotSTFT(self, log=True):
		""" Plot the spectrogram or the log-spectrogram.

		Parameters
		----------
		log : bool
			The fonction plots the log-spectrogram if it's True, and the classic
			spectrogram else.
		"""
		if log:
			if self.STFTlog is None:
				self.computeSTFTlog()
			plt.figure()
			#plt.imshow(np.sqrt(self.STFT), origin='lower', aspect='auto', extent=[self.STFTsec[0], self.STFTsec[-1], self.STFTfreq[0], self.STFTfreq[-1]], interpolation='nearest')
			plt.pcolormesh(self.STFTsec, np.arange(0, 128), np.sqrt(self.STFTlog))
			plt.ylim((0, 128))
			plt.xlabel('Time(s)')
			plt.ylabel('Fréquency bin')
		else:
			if self.STFT is None:
				self.computeSTFT()
			plt.figure()
			#plt.imshow(np.sqrt(self.STFT), origin='lower', aspect='auto', extent=[self.STFTsec[0], self.STFTsec[-1], self.STFTfreq[0], self.STFTfreq[-1]], interpolation='nearest')
			plt.pcolormesh(self.STFTsec, self.STFTfreq, np.sqrt(self.STFT))
			plt.ylim((30, 6000))
			plt.xlabel('Time(s)')
			plt.ylabel('Fréquency(Hz)')
		plt.show()

	def computeCQT(self, nbins=128):
		vect = self.data[:,]
		vect = vect[:,0]
		self.CQT = np.abs(librosa.cqt(vect, sr=self.sampleRate, fmin=30, n_bins=nbins, bins_per_octave=16))

	def getCQT(self, nbins=128):
		""" Returns the Constant Quality Factor, computed with computeCQT.

		Parameters
		----------
		nbins : int
			Number of frequency bins of the resulting CQT. Affects the Frequency
			resolution.
		"""

		if self.CQT is None:
			self.computeCQT(nbins)
		return self.CQT

	def plotCQT(self):
		""" Plots the CQT. Compute it if necessary."""
		if self.CQT is None:
			self.computeCQT()

		plt.figure()
		plt.pcolormesh(np.arange(0,len(self.CQT[0,:])), np.arange(0,len(self.CQT[:,0])), self.CQT)
		plt.colorbar(format='%+2.0f dB')
		plt.tight_layout()
		plt.xlabel('Time')
		plt.ylabel('Frequency bin')
		plt.show()
