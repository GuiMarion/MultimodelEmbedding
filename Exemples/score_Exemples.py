import sys 
sys.path.append('../')

from Modules import score
from Modules import waveForm

def isThereVelocity(s):	
	for elem in s.getPianoRoll():
		for e2 in elem:
			if e2 != 0 and e2 != 1.0:
				return True

	return False



# Import one of my masterpieces ...
s = score.score(pathToMidi="Gui's_score.mid", velocity=True)

print("The length of the file is ",s.length, "seconds.")

print("Is the object a part of a midi file ?", s.ispart)

print("Is the object handle velocity ?", isThereVelocity(s))

# Plot the piano roll representation of the score
s.plot()

# print the pianoRoll matrix
print(s.getPianoRoll())

# plot the 10th first beats
sub = s.extractPart(0, 10)
sub.plot()

print("Is the object a part of a midi file ?", sub.ispart)

quit()

# play the waveform of the piece
w = s.toWaveForm()
w.play()
