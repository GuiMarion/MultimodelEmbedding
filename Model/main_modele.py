import sys 
sys.path.append('../')

from Model import modele

import torch


EPOCHS = 30

model = modele.Modele("../DataBaseTest", gpu=1)

model.learn(EPOCHS, learning_rate=1e-1)
