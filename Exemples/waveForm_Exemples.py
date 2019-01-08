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

# play the waveform of the piece
w = waveForm.waveForm("Gui's_song.wav")

# plot the signal
w.plot()

# plt the FFT
w.plotFFT()

# get the STFT
w.getSTFT()

#plot the STFT
w.plotSTFT()

#get the log-frequency spectrogram
w.getSTFTlog()

#plot the log-STFT
w.plotSTFTlog()

# create a file testouille containing the waveform
w.save("testouille.wav")
