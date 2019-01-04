import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm

# Import one of my masterpieces ...
#s = score.score("Gui's_score.mid")
s = score.score("velocity.mid")

# Plot the piano roll representation of the score
s.plot()

quit()

# print the pianoRoll matrix
print(s.getPianoRoll())

# plot the 10th first beats
sub = s.extractPart(0, 10)
sub.plot()

# play the waveform of the piece
w = s.toWaveForm()
w.play()
