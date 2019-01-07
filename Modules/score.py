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
	def __init__(self, pathToMidi, velocity=False, quantization=24, fromArray=(None, "")):


		if fromArray[0] is None:
			try:
				# use pypianoroll to parse the midifile
				self.pianoroll = proll(pathToMidi,beat_resolution=quantization).get_merged_pianoroll()
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
		self.length = len(self.pianoroll)//16

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll

		return self.pianoroll

	def getLength(self):
		# return the length in tim beat

		return self.length

	def plot(self):
		# plot the pianoRoll representation
		
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

		for i in range(N//step - windowSize):
			retParts.append(self.extractPart(i*step, i*step+windowSize))

		return retParts

	def toWaveForm(self, font="000_Florestan_Piano.sf2"):

		midiPath = ".TEMP/"+self.name+".mid"
		wavePath = ".TEMP/"+self.name+".wav"
		pathFont = "../SoundFonts/" + font

		self.writeToMidi(midiPath)
		process = subprocess.Popen("fluidsynth -F "+wavePath+" "+pathFont+" "+midiPath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()
		# should return on an object of type waveForm defined in this folder
		newWaveForm = waveForm.waveForm(wavePath)

		# cleaning the temporary files
		process = subprocess.Popen("rm -f " + midiPath + " " + wavePath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()

		return newWaveForm

	def getTransposed(self):
		# Should return a list of 12 scores corresponding to the 12 tonalities.

		transposed_scores = []
		
		# Transposes from 6 semitones down to 5 semitones up
		# And stores each transposition as a new score
		for t in range(-6, 6):
			transRoll = shift(self.pianoroll, t) # transposed piano roll matrix
			newName = self.name + '_' + str(t)
			
			transposed_score = score("", fromArray=(transRoll, newName))
			transposed_scores.append(transposed_score)

		return transposed_scores
		
	def writeToMidi(self, midiPath):
		tempTrack = Track(pianoroll=self.pianoroll, program=0, is_drum=False,
									name=self.name)
		tempMulti = proll(tracks=(tempTrack,), beat_resolution=self.quantization)
		tempMulti.write(midiPath)

def shift(mat, t):
	# Vertically shifts a matrix by t rows.
	# Shifts up if t is positive, down if t is negative.
	# Fills empty slots with zeros.
	
    result = np.empty_like(mat)
    if t < 0:
        result[:-t] = 0
        result[-t:] = mat[:t]
    elif t > 0:
        result[-t:] = 0
        result[:-t] = mat[t:]
    else:
        result = mat
        
    return result
	
