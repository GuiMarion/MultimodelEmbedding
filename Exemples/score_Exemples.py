import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm
from pypianoroll import transpose

# Import one of my masterpieces ...
#s = score.score("chp_op18.mid")
#s = score.score("velocity.mid")
s = score.score("kikou.mid")
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


"""

list = s.pyRoll.tranpose(2)
