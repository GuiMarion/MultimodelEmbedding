import sys 
sys.path.append('../')

from Model import modele

import torch


EPOCHS = 1

model = modele.Modele("../DataBaseTest/", gpu=0, outPath=".TEMP")

model.learn(EPOCHS, learning_rate=1e-2)
