import sys 
sys.path.append('..')

from Modules import score
from Modules import waveForm

'''
	You can create instanciate from a score object
'''


'''
	Or also directly from a file
'''

w = waveForm.waveForm("Gui's_song.wav")

# play the waveform of the piece
w.play()

# plot the signal
w.plot()

# plt the FFT
w.plotFFT()

print(w.getFFT())

# create a file testouille containing the waveform
w.save("testouille.wav")


quit()

# Import one of my masterpieces ...
s = score.score("Gui's_song.mid")

# play the waveform of the piece
w = s.toWaveForm()
w.play()

# plot the signal
w.plot()

# plt the FFT
w.plotFFT()

print(w.getFFT())

# create a file testouille containing the waveform
w.save("testouille.wav")
