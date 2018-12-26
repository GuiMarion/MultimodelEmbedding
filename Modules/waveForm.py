import sounddevice as sd

class waveForm:
	def __init__(self, path):

		if path is not None:
			self.loadFromFile(path)

		self.name = ""
		self.tempo = 0
		self.sampleRate = 0
		self.data = []
		
	def loadFromFile(self, path):
		# update the attributes, should use loadFrom data

		return "TODO"

	def loadFromData(self, sequence):
		# update the attributes
		# input sequence is a vector of samples
		# should work with mono as well as with stereo 
		return "TODO"

	def play(self):
		# play the data, use the module sound device

		fs = self.sampleRate
		return sd.play(myarray, fs)

	def save(self, path):
		# save as a wav file at path
		return "TODO"

	def plot(self):
		# plot the signal
		return "TODO"

	def getFFT(self):
		# return FFT if already computed or compute it, store it and return it
		# store the result of the FFT as attribute in order to compute it only once
		return "TODO"

	def plotFFT(self):
		# plot FFT
		return "TODO"