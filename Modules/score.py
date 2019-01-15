from Modules import waveForm

from pypianoroll import Multitrack as proll
from pypianoroll import Track
try:
	import matplotlib.pyplot as plt
	plot = True
except ImportError:
	plot = False
import subprocess
import os
import numpy as np
import copy
from midi2audio import FluidSynth

import sys

SERVER = True

class NullWriter(object):
	def write(self, arg):
		pass
'''
velocity : ok
getpianoroll : ok
plot : ok
length(in timebeat) : pas ok
extract part: ok
towaveform : ok
transpose : ok
'''

'''
TODO :
	- Test compatibility with other modules (database, waveform)
	- Resoudre probleme de tempo
'''

class score:
	def __init__(self, pathToMidi, velocity=False, quantization=24, fromArray=(None, "")):


		if fromArray[0] is None:
			try:
				# use pypianoroll to parse the midifile
				self.pianoroll = proll(pathToMidi, beat_resolution=quantization)
				self.pianoroll.trim_trailing_silence()
				self.pianoroll = self.pianoroll.get_merged_pianoroll()
				self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
			except OSError:
				raise RuntimeError("incorrect midi file.")

		else:
			self.name = fromArray[1]
			self.pianoroll = fromArray[0]

		# store the numpy array corresponding to the pianoroll
		self.velocity = velocity
		self.quantization = quantization

		#store length in time beat
		self.length = len(self.pianoroll)//self.quantization

		self.transposition = 0

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll

		return np.transpose(self.pianoroll)

	def getLength(self):
		# return the length in time beat

		return self.length

	def plot(self):
		# plot the pianoRoll representation
<<<<<<< HEAD
=======
		if plot == False:
			print("you cannot plot anything as matplotlib is not available")
			return
>>>>>>> 2d8e66547586195e097e1c6a0d957ffaea2701f4

		plt.imshow(self.pianoroll.T, aspect='auto', origin='lower')
		plt.xlabel('time (beat)')
		plt.ylabel('midi note')
		plt.grid(b=True, axis='y')
		plt.yticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
           ["C-2", "C-1", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
		color = plt.colorbar()
		color.set_label('velocity', rotation=270)
		plt.show()

	def extractPart(self, start, end, inBeats=False):

		# return a score object including this one between start and end in time beat
		if inBeats is True:
			if start >= 0 and end < self.length:
				pianoRollPart = self.pianoroll[start*self.quantization : end*self.quantization, : ]
				newName = self.name+ "_" + str(start) + "_" + str(end)

				scorePart = score("", fromArray=(pianoRollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")
		else:
			if start >= 0 and end < self.length*self.quantization:
				pianoRollPart = self.pianoroll[start : end, : ]
				newName = self.name+"_" + str(start) + "_" + str(end)

				scorePart = score("", fromArray=(pianoRollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")


	def extractAllParts(self, length, step=1):
		# Extract all parts of size length beats
		N = self.length*self.quantization
		windowSize = length*self.quantization
		retParts = []

		for i in range((N-windowSize)//step):
			retParts.append(self.extractPart(i*step, i*step+windowSize))

		return retParts

	def toWaveForm(self, font="MotifES6ConcertPiano.sf2"):

		if SERVER == True:
			midiPath = "/fast-1/guilhem/"+self.name+".mid"
			wavePath = "/fast-1/guilhem/"+self.name+".wav"
		else:
			midiPath = ".TEMP/"+self.name+".mid"
			wavePath = ".TEMP/"+self.name+".wav"

		pathFont = "../SoundFonts/" + font

		self.writeToMidi(midiPath)
		process = subprocess.Popen("fluidsynth -F "+wavePath+" "+pathFont+" "+midiPath, shell=True,
									stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()
		nullwrite = NullWriter()
		oldstdout = sys.stdout
		oldstderr = sys.stderr
		sys.stdout = nullwrite # disable output
		sys.stderr = nullwrite

		F = FluidSynth(pathFont)
		F.midi_to_audio(midiPath, wavePath)

		sys.stdout = oldstdout # enable output
		sys.stderr = oldstderr

		# should return on an object of type waveForm defined in this folder
		newWaveForm = waveForm.waveForm(wavePath)

		# cleaning the temporary files
		process = subprocess.Popen("rm -f " + midiPath + " " + wavePath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()

		return newWaveForm

	def transpose(self, t):

		# Vertically shifts a matrix by t rows.
		# Fills empty slots with zeros.

	    result = np.empty_like(self.pianoroll)
	    if t > 0:
	        result[:,:t] = 0
	        result[:,t:] = self.pianoroll[:,:-t]
	    elif t < 0:
	        result[:,t:] = 0
	        result[:,:t] = self.pianoroll[:,-t:]
	    else:
	        result = self.pianoroll

	    return result

	def getTransposed(self):
		# Should return a list of 12 scores corresponding to the 12 tonalities.

		transposed_scores = []

		# Transposes from 6 semitones down to 5 semitones up
		# And stores each transposition as a new score
		for t in range(-6, 6):
			transRoll = self.transpose(t) # transposed piano roll matrix
			newName = self.name + '_' + str(t) + "_"

			transposed_score = score("", fromArray=(transRoll, newName))
			transposed_score.transposition = t
			transposed_scores.append(transposed_score)

		return transposed_scores

	def aumgmentData(self):
		# function that do data augmentation

		data = getTransposed

		return data


	def writeToMidi(self, midiPath):
		tempTrack = Track(pianoroll=self.pianoroll, program=0, is_drum=False,
									name=self.name)
		tempMulti = proll(tracks=(tempTrack,), beat_resolution=self.quantization)
		tempMulti.write(midiPath)
