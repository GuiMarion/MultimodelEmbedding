import sys
import time
sys.path.append('..')

from Modules import score
from Modules import waveForm


'''
	You can instanciate from a score object and extract parts from it.
	Then you can create associate waveForm objects.
'''

s = score.score('dataBaseTest/MIDIs/xmas/bk_xmas3.mid')
s1 = s.extractPart(0, 5)
print(s1)
w = []
#for midipart in s1:
s1.writeToMidi('dataBaseTest/scoreTest/'+s1.name+".mid")
#w.append(midipart.toWaveForm())
w = s1.toWaveForm()
print(w)

'''
	Or also directly from a file
'''
STFTarrays = []
CQT = []

t1 = time.time()
#for wavepart in w:STFTarrays.append(
w.getSTFTlog()
t2 = time.time()
t3 = time.time()
#for wavepart in w:CQT.append(
w.getCQT()
t4 = time.time()

dt1 = t2 - t1
dt2 = t4 - t3
print('STFT: ' + str(dt1) + ' sec')
print('CQT: ' + str(dt2) + ' sec')
