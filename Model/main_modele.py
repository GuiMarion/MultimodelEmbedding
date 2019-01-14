import modele
import torch
import network

EPOCHS = 30

model = modele.Modele("../DataBaseTest", gpu=2)

model.learn(EPOCHS, learning_rate=1e-1)
