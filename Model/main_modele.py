import modele
import torch
import network

EPOCHS = 50

model = modele.Modele("dataBaseTest.data")

model.learn(EPOCHS, learning_rate=1e-1)