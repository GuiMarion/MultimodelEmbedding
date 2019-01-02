import network
import torch

# create a complete CNN
model = network.Net()
x = torch.randn(2, 1, 80, 100)
y = torch.randn(2, 32)

model.learn(x, y, 1000)


