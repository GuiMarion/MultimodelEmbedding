from Modules import waveForm

class score:
	def __init__(self, pahtToMidi, velocity=False):

		# Attributs to fill from the file
		self.name = ""
		self.tempo = 0
		# compute the pianoroll with computePianoRoll, just also work with velocity = True
		self.pianoRoll = ""
		self.instrument = ""
		self.lenght = 0

	def computePianoRoll(self, velocity=False):

		return "TODO"

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll
		return "TODO"

	def plot(self):
		# plot the pianoRool representation
		print("TODO")
		return "TODO"

	def toWaveForm(self, sound="default"):

		# should return on an object of type waveForm defined in this folder
		return waveForm.waveForm(None)

	def extractPart(self, begin, lenght="default"):

		# should return a temporal part of pianoroll as a score object
		return score("")
