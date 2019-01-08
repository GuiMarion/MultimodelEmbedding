import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm
from Modules.score import shift
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import one of my masterpieces ...
#s = score.score("velocity.mid")
s = score.score("testFelix.mid")
# ~ print(s.getLength())
# ~ aaa = s.extractAllParts(10, step=100)
# ~ print(len(aaa))
# ~ aaa[0].plot()

# ~ a = s.extractAllParts(10, step=600)
# ~ print(len(a))
# ~ a[12].plot()

print(s.pianoroll.shape)

# ~ x = s.getTransposed()
# ~ for transpy in x:
	# ~ transpy.plot()

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


