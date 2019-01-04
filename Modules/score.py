from Modules import waveForm

from pypianoroll import Multitrack as proll

import matplotlib.pyplot as plt
import os

class score:
	def __init__(self, pathToMidi, velocity=False):

		# Attributs to fill from the file
		self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
		# compute the pianoroll with computePianoRoll, just also work with velocity = True
		self.pyRoll = proll(pathToMidi)
		self.velocity = velocity

		if velocity is True:
			pyRoll.binarize()


		self.tempo = 0
		self.length = 0

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll

		return self.pyRoll.get_merged_pianoroll()

	def plot(self):
		# plot the pianoRool representation
		
		self.pyRoll.plot()
		plt.show()

	def toWaveForm(self, sound="default"):

		# should return on an object of type waveForm defined in this folder
		return waveForm.waveForm(None)

	def extractPart(self, begin, lenght="default"):

		# should return a temporal part of pianoroll as a score object
		return score("")
