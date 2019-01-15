import sys 
sys.path.append('../')

from Model import modele

import torch


EPOCHS = 50

model = modele.Modele("../DataBase/MIDIs/mozart", gpu=1)

model.learn(EPOCHS, learning_rate=1e-2)
