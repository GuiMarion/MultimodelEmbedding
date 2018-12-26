from Modules import waveForm
from music21 import converter
import math
import numpy as np
import os

class score:
	def __init__(self, pathToMidi, velocity=False):

		# Attributs to fill from the file
		self.name = ""
		self.tempo = 0
		# compute the pianoroll with computePianoRoll, just also work with velocity = True
		self.pianoRoll = ""
		self.instrument = ""
		self.lenght = 0
		#convert into a music21 stream
		self.piece = converter.parse(pathToMidi) #convert to music 21 stream

		


	def computePianoRoll(self, velocity=False):

		piece = self.piece	#extract piece music21 stream from score 
		quantization = 16  #set the quantization number of notes / s

		# Extract the parts from the piece and compute piano roll on for each part
		all_parts = {}
		for part in piece.parts:
			print(part)
			try:
				track_name = part[0].bestName()
			except AttributeError:
				track_name = 'None'

		
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
				t_end = int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset + el.duration.quarterLength)*quantization))
				if(t_end>duration_max):
					duration_max=t_end
			# Get the pitch and offset+duration
			piano_roll_part = np.zeros((128,math.ceil(duration_max)))

			for this_note in part.recurse(classFilter=('Note')):
				note_start = int(math.ceil(((measure_offset[this_note.measureNumber] or 0) + this_note.offset)*quantization))
				note_end = int(math.ceil(((measure_offset[this_note.measureNumber] or 0) + this_note.offset + this_note.duration.quarterLength)*quantization))
				piano_roll_part[int(this_note.pitch.ps),note_start:note_end] = 1
			#
			cur_part = piano_roll_part;
			if (cur_part.shape[1] > 0):
				all_parts[track_name] = cur_part;

			return piece, all_parts

	def getPianoRoll(self):
		# return the np.array containing the pianoRoll
		piece, all_parts = self.computePianoRoll(velocity=False)
		return all_parts

	def plot(self):
		# plot the pianoRool representation
		piece, all_parts = self.computePianoRoll(velocity=False)
		return piece.plot()

	def toWaveForm(self, sound="default"):

		# should return on an object of type waveForm defined in this folder
		return waveForm.waveForm(None)

	def extractPart(self, begin, lenght="default"):

		# should return a temporal part of pianoroll as a score object
		return score("")

