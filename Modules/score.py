import math
import numpy as np
import re
import subprocess
import os
import platform
import matplotlib.pyplot as plt
from pypianoroll import Track, Multitrack, write
from music21 import converter

class score:
	def __init__(self, pathToMidi="", velocity=False, quantization=16, pianoRoll=None, name=""):

		# is true if the object is a part of a midi file
		self.ispart = pianoRoll is not None

		# quantization of the pianoRoll
		self.quantization = quantization

		# get the piano roll as a numpy array
		if pianoRoll is None:
			self.pianoRoll = self.computePianoRoll(pathToMidi, velocity)
			self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
		else:
			self.pianoRoll = pianoRoll
			self.name = name

		# duration in seconds
		self.length = self.pianoRoll.shape[0] // self.quantization # Why it dosent work ?


	def computePianoRoll(self, pathToMidi, velocity):
		# Extract the parts from the piece and compute piano roll on for each part
		piece = converter.parse(pathToMidi)
		all_parts = {}
		k = 0
		for part in piece.parts:
			k += 1
			try:
				track_name = part[0].bestName()
			except AttributeError:
				track_name = 'Track ' + str(k)


			# Get the measure offsets
			measure_offset = {None:0}
			for el in part.recurse(classFilter=('Measure')):
				measure_offset[el.measureNumber] = el.offset
			# Get the measure offsets
			measure_offset = {None:0}
			for el in part.recurse(classFilter=('Measure')):
				measure_offset[el.measureNumber] = el.offset
			# Get the duration of the part
			duration_max = 0
			for el in part.recurse(classFilter=('Note','Rest')):
				t_end = int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset + el.duration.quarterLength) * self.quantization))
				if(t_end>duration_max):
					duration_max=t_end
			# Get the pitch and offset+duration
			piano_roll_part = np.zeros((128,math.ceil(duration_max)))

			for this_note in part.recurse(classFilter=('Note')):
				note_start = int(math.ceil(((measure_offset[this_note.measureNumber] or 0) + this_note.offset) * self.quantization))
				note_end = int(math.ceil(((measure_offset[this_note.measureNumber] or 0) + this_note.offset + this_note.duration.quarterLength) * self.quantization))
				piano_roll_part[int(this_note.pitch.ps),note_start:note_end] = 1
			#
			cur_part = piano_roll_part;
			if (cur_part.shape[1] > 0):
				all_parts[track_name] = cur_part;

			rollArray = np.array(list(all_parts.values()))
			return rollArray[0].T

	def getPianoRoll(self):
		# return the pianoroll as an np array

		return self.pianoRoll

	def extractPart(self, start, end):
		# return the piano roll between start and end in seconds

		if start >= 0 and end*self.quantization < self.pianoRoll.shape[0]:
			pianoRollPart = self.pianoRoll[start*self.quantization : end*self.quantization, : ]
			newName = self.name+"_" + str(start) + "_" + str(end)
			scorePart = score(pianoRoll=pianoRollPart, name=newName)

			return scorePart
		else:
			raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")


	def plot(self):
	    # plot the pianoRoll representation
		track = Track(pianoroll = self.pianoRoll, program = 0, is_drum = False, name='track')
		fig, ax = track.plot()
		plt.show()


	def toWaveForm(self, sound="default"):
		# Write a WAV file from midi using musescore
		if platform.system() == 'Linux':
			musepath = str(subprocess.check_output(["which", "mscore"]))
			musepath = musepath[1:-3] + "'"
		cmd = musepath + ' ' + '\"' + self.path + '\"' + " -o " + '\"' + re.sub(".mid", ".wav", self.path) + '\"'
		subprocess.call(cmd, shell=True)
		return os.path.splitext(os.path.basename(self.path))[0] + '.wav'