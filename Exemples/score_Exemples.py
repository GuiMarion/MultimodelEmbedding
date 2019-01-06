import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm
import matplotlib.pyplot as plt
import copy

# Import one of my masterpieces ...
#s = score.score("chp_op18.mid")
#s = score.score("velocity.mid")
s = score.score("chp_op18.mid")
"""
'''
Part for the midi
'''

# Return the length in time beat
print("length:", s.getLength())

# Plot the piano roll representation of the score
s.plot()

# print the pianoRoll matrix
print(s.getPianoRoll())

# play the waveform of the piece
w = s.toWaveForm()
w.play(5)

# Extract all parts of size 1 beat with a step of 10 samples
L = s.extractAllParts(1, step= 10)
for elem in L:
	elem.plot()


'''
Part for the extractPart
'''


# plot the 10th first beats
sub = s.extractPart(0, 10)
sub.plot()

# print the pianoroll matrix
print(sub.getPianoRoll())

# Return the length in time beat
print("length:", sub.getLength())

# play the waveform of the 10th first beats
w = s.toWaveForm()
w.play(5)


sub = s.extractPart(0, 10)
sub.plot()
plt.show()
sub.pyRoll.transpose(2)

sub.plot()
plt.show()

"""

transposed_pianorolls = []



sub = s.extractPart(0,30, inBeats=True)

sub.plot()

sub1 = sub.getTransposed()

for elem in sub1:
	elem.plot()
