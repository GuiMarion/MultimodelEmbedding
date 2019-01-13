import sys
import time
import pylab
sys.path.append('..')

from Modules import score
from Modules import waveForm


'''
	You can instanciate from a score object and extract parts from it.
	Then you can create associated waveForm objects.
'''

s = score.score('dataBaseTest/MIDIs/xmas/bk_xmas3.mid')
s1 = s.extractAllParts(5, step=1000)
print(s1)

w = []
for midipart in s1:
	midipart.writeToMidi('../'+midipart.name+".mid")
	w.append(midipart.toWaveForm())
print(w)

STFTarrays = []
CQT = []

t1 = time.time()
for wavepart in w:
	STFTarrays.append(wavepart.getSTFTlog())
t2 = time.time()
t3 = time.time()
for wavepart in w:
	CQT.append(wavepart.getCQT())
t4 = time.time()

# w.plotSTFTlog()
# w.plotCQT()

dt1 = t2 - t1
dt2 = t4 - t3

print('STFT: ' + str(dt1) + ' sec')
print('CQT: ' + str(dt2) + ' sec')
