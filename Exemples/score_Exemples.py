import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm

# Import one of my masterpieces ...
s = score.score("chp_op18.mid")
#s = score.score("velocity.mid")

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