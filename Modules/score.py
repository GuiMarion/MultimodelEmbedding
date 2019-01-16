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
<<<<<<< HEAD
=======
'''
velocity : ok
getpianoroll : ok
plot : ok
length(in timebeat) : pas ok
extract part: ok
towaveform : ok
transpose : ok
'''
>>>>>>> a3be14b5ba997a3daf301bbc5053b1510cc0b9d3

'''
TODO :
	- Test compatibility with other modules (database, waveform)
	- Resoudre probleme de tempo
'''

class score:
	"""This class is used to manage midi data from the dataset.

	It uses the module pypianoroll, that allows the managing of numpy arrays
	instead of raw midi files. Thus, score class can be defined from a midi
	file, but also from a pypianoroll object extracted from a midi (see
	extractPart and extractAllParts). Midi or pianoroll numpy arrays excerpts
	can be cropped, plotted, transposed and written as new midi excerpts, and
	even converted into waveForm objects.

	Attributes
	----------
	self.pianoroll : :obj:'list' of :obj:'int'
		A list of numpy arrays that represents the midi data.
	self.name : str
		If a score class is created from a raw midi file, this attributes is
		corresponding to the name of the midi file without a '.mid' or '.midi'
		extension. If it is created from another score class (see extractPart
		and extractAllParts), a new name is created, that indicates wich excerpt
		of the midi file it corresponds to.
	self.velocity : bool
		False by default. Indicates if midi velocity is computed from the midi
		file, or not.
	self.quantization : int
		MIDI quantization per beat. 24 by default.
	self.length : int
		Beat duration of the midi data.
	self.transposition : int
		Indicates if the data is transposed from another score object. In
		semitones.
	"""
	def __init__(self, pathToMidi, velocity=False, quantization=24, fromArray=(None, "")):
		"""Initialises a score object.

		All the attributes of the class are computed from it.

		Parameters
		----------
		pathToMidi : str
			Path of the midi data associated to the class. Used only if a score
			object is defined from a raw midi file.
		velocity : bool
			Initializes self.velocity. False by default.
		quantization : int
			Initializes self.quantization. 24 by default.
		fromArray : :obj: 'list' of :obj: 'str'
			By default, indicates that the score object is defined from a raw
			midi file. If the score object is defined from extractPart or
			extractAllParts, contains the name and the pianoroll of the excerpt,
			that initializes self.name and self.pianoroll.
		"""

		if fromArray[0] is None:
			try:
				# use pypianoroll to parse the midifile
				self.pianoroll = proll(pathToMidi, beat_resolution=quantization)
				self.pianoroll.trim_trailing_silence()
				self.pianoroll = self.pianoroll.get_merged_pianoroll()
				self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
			except OSError:
				raise RuntimeError("Incorrect midi file.")
		else:
			self.name = fromArray[1]
			self.pianoroll = fromArray[0]

		self.velocity = velocity
		self.quantization = quantization
		self.length = len(self.pianoroll)//self.quantization
		self.transposition = 0

	def getPianoRoll(self):
		"""Return the np.array containing the pianoRoll."""

		return np.transpose(self.pianoroll)

	def getLength(self):
		"""Return the length in time beat."""

		return self.length

	def plot(self):
		"""Plot the pianoRoll representation."""

		if plot == False:
			print("you cannot plot anything as matplotlib is not available")
			return

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
		"""Extract excerpt from data, and returns a score object from it.

		Parameters
		----------
		start : int
			start of the desired excerpt. Can be expressed in quantisized or
			unquantisized beats.
		end : int
			end of the desired excerpt. Must be expressed in the same dimension
			as start.
		inBeats : bool
			Indicates if start and end parameters are quantisized, or not. False
			by default.
		"""
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
		"""Extract parts of a given length and step size, through all the data.

		Notice that it uses extractPart function for each excerpt, thus it
		creates score objects.

		Parameters
		----------
		length : int
			Length of desired excerpts in beat duration (unquantisized).
		step : int
			Length of the step between each excerpt in the file. 1 by default.
		"""

		N = self.length*self.quantization
		windowSize = length*self.quantization
		retParts = []

		for i in range((N-windowSize)//step):
			retParts.append(self.extractPart(i*step, i*step+windowSize))

		return retParts

	def toWaveForm(self, font="MotifES6ConcertPiano.sf2"):
		""" Converts data into a waveForm object.

		It uses fluidsynth module to perform audio conversion. Notice that a
		temporary wave sound file is created, which then is immediately erased.
		Also, SERVER is a hypercriterion that indicates if the converted data
		has to be saved into a given path.

		Parameters
		----------
		font : str
			Path to a soundfont which is used by fluidsynth to create audio
			data from midi. "MotifES6ConcertPiano.sf2" by default.
		"""

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
		"""Return a transposed pianoroll.

<<<<<<< HEAD
		Parameters
		----------
		t : int
			Value of transposition, in semitones.
		"""
=======
		# Vertically shifts a matrix by t rows.
		# Fills empty slots with zeros.

>>>>>>> a3be14b5ba997a3daf301bbc5053b1510cc0b9d3
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
		"""Returns a list of 12 transposed score objects.

		Notice that the returned list contains all the possible transpositions
		(in an octave range) from midi data.
		"""
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
		"""Function that performs data augmentation from initial data."""

		data = getTransposed

		return data


	def writeToMidi(self, midiPath):
		"""Saves pianoroll data into midi data, in a given path.

		Parameters
		----------
		midiPath : str
			Path were the corresponding midi data is saved.
		"""
		tempTrack = Track(pianoroll=self.pianoroll, program=0, is_drum=False,
									name=self.name)
		tempMulti = proll(tracks=(tempTrack,), beat_resolution=self.quantization)
		tempMulti.write(midiPath)
