from pypianoroll import Multitrack as proll
import matplotlib.pyplot as plt


roll = proll("velocity.mid")

plt.plot(roll.get_merged_pianoroll())
plt.show()

#proll.write(roll, "prout.mid")