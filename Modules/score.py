from Modules import waveForm

from pypianoroll import Multitrack as proll
from pypianoroll import Track

import matplotlib.pyplot as plt
import subprocess
import os
import numpy as np
import copy

'''
velocity : ok 
getpianoroll : ok
plot : plot parts separatly but it doesn't matter
length(in timebeat) : ok
extract part: ok 
towaveform : ok
tranpose : no

'''

'''
TODO :
	- Tranpose
	- Check midi validity and raise an error
'''

class score:
	def __init__(self, pathToMidi, velocity=False, quantization=24, frompyRoll=(None, "")):


		if frompyRoll[0] == None:
			try:
				# use pypianoroll to parse the midifile
				self.pyRoll = proll(pathToMidi,beat_resolution=quantization)
				self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
				self.pianoroll = self.pyRoll.get_merged_pianoroll()
			except OSError:
				raise RuntimeError("incorrect midi file.")

		else:
			self.pyRoll = frompyRoll[0]
			self.name = frompyRoll[1]
			self.pianoroll = self.pyRoll.get_pianoroll_copy()

		# store the numpy array corresponding to the pianoroll
		self.velocity = velocity
		self.quantization=quantization

		if velocity is False:
			self.pyRoll.binarize()

		#store length in time beat
		self.length = len(self.pianoroll)//16

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll

		return self.pianoroll

	def getLength(self):
		# return the length in tim beat

		return self.length

	def plot(self):
		# plot the pianoRool representation
		
		self.pyRoll.plot()
		plt.show()

	def extractPart(self, start, end, inBeats=False):

		# return a score object including this one between start and end in time beat
		if inBeats is True:
			if start >= 0 and end < self.length:
				pianoRollPart = self.pianoroll[start*self.quantization : end*self.quantization, : ]
				newName = self.name+"_" + str(start) + "_" + str(end)

				pyrollPart = Track(pianoroll=pianoRollPart, program=0, is_drum=False,
	              name=newName)
				
				scorePart = score("", frompyRoll=(pyrollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")
		else:
			if start >= 0 and end < self.length*self.quantization:
				pianoRollPart = self.pianoroll[start : end, : ]
				newName = self.name+"_" + str(start) + "_" + str(end)

				pyrollPart = Track(pianoroll=pianoRollPart, program=0, is_drum=False,
	              name=newName)
				
				scorePart = score("", frompyRoll=(pyrollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")		


	def extractAllParts(self, length, step=1):
		# Extract all parts of size length beats
		N = self.length*self.quantization
		windowSize = length*self.quantization
		retParts = []

		for i in range(N//step - windowSize):
			retParts.append(self.extractPart(i*step, i*step+windowSize))

		return retParts

	def toWaveForm(self, font="000_Florestan_Piano.sf2"):

		midiPath = ".TEMP/"+self.name+".mid"
		wavePath = ".TEMP/"+self.name+".wav"
		pathFont = "../SoundFonts/" + font

		self.pyRoll.write(midiPath)
		process = subprocess.Popen("fluidsynth -F "+wavePath+" "+pathFont+" "+midiPath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()
		# should return on an object of type waveForm defined in this folder
		newWaveForm = waveForm.waveForm(wavePath)

		# cleaning the temporary files
		process = subprocess.Popen("rm -f " + midiPath + " " + wavePath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()

		return newWaveForm

	def getTransposed(self):
		# should return a list of 12 objects corresponding au 12 tonalities.
		# the algorithm should make a good choice in up-tranposing or down-tranposing
		# for exemple if the piece is very high we will down-tranpose.

		if not isinstance(self.pyRoll, Track):

			raise TypeError("Can only transpose Track objects")


		transposed_pianorolls = []
		range_pianoroll = self.pyRoll.get_active_pitch_range() # return piano roll pitch range

		if np.abs(range_pianoroll[0]-69) < np.abs(range_pianoroll[1]-69): # compare piano roll pitch range with note A4

			# down-transposing

			for tonality in range(12):
				pyRoll_temp = copy.deepcopy(self.pyRoll)
				pyRoll_temp.transpose(-tonality)
				transposed_pianorolls.append(pyRoll_temp)

		else:

			# up-transposing

			for tonality in range(12):
				pyRoll_temp = copy.deepcopy(self.pyRoll)
				pyRoll_temp.transpose(+tonality)
				transposed_pianorolls.append(pyRoll_temp)

		return transposed_pianorolls




