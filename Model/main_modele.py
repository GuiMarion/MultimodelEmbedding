import modele
import torch
import network

EPOCHS = 2

model = modele.Modele("../DataBaseTest", gpu=None)

model.learn(EPOCHS, learning_rate=1e-1)
