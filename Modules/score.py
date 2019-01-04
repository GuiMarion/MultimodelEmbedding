from Modules import waveForm

from pypianoroll import Multitrack as proll
from pypianoroll import Track

import matplotlib.pyplot as plt
import subprocess
import os

'''
velocity : ok 
getpianoroll : ok
plot : plot parts separatly but it doesn't matter
length(in timebeat) : ok
extract part: no 
towaveform : no
tranpose : no

'''

class score:
	def __init__(self, pathToMidi, velocity=False, quantization=24, frompyRoll=(None, "")):


		if frompyRoll[0] == None:
			# use pypianoroll to parse the midifile
			self.pyRoll = proll(pathToMidi,beat_resolution=quantization)
			self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
			self.pianoroll = self.pyRoll.get_merged_pianoroll()

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

	def extractPart(self, start, end):

		# return a score object including this one between start and end in time beat

		if start >= 0 and end < self.length:
			pianoRollPart = self.pianoroll[start*self.quantization : end*self.quantization, : ]
			newName = self.name+"_" + str(start) + "_" + str(end)

			pyrollPart = Track(pianoroll=pianoRollPart, program=0, is_drum=False,
              name=newName)
			
			scorePart = score("", frompyRoll=(pyrollPart, newName))

			return scorePart
		else:
			raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")

	def toWaveForm(self, pathFont="../SoundFonts/000_Florestan_Piano.sf2"):

		midiPath = ".TEMP/"+self.name+".mid"
		wavePath = ".TEMP/"+self.name+".wav"

		self.pyRoll.write(midiPath)
		process = subprocess.Popen("fluidsynth -F "+wavePath+" "+pathFont+" "+midiPath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()
		# should return on an object of type waveForm defined in this folder
		newWaveForm = waveForm.waveForm(wavePath)

		# cleaning the temporary files
		process = subprocess.Popen("rm -f " + midiPath + " " + wavePath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()

		return newWaveForm




