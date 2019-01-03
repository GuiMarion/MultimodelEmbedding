import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm

# Import one of my masterpieces ...
s = score.score(pathToMidi="Gui's_score.mid")

print(s.length)

# Plot the piano roll representation of the score
s.plot()

# print the pianoRoll matrix
print(s.getPianoRoll())

print(s.getPianoRoll().shape)

# plot the 10th first beats
sub = s.extractPart(0, 10)
sub.plot()

quit()

# play the waveform of the piece
w = s.toWaveForm()
w.play()
