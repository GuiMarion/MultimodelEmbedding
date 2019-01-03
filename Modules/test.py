from soundfile import SoundFile
import soundfile as sf

#myFile = SoundFile("Gui's_song.wav")
data, sample_rate = sf.read("Gui's_song.wav")
#namefile = myFile.samplerate()
print (sample_rate)