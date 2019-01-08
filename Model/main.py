import network
import torch

N = 10
EPOCHS = 50
# create a complete CNN
model = network.Net()
x = torch.randn(N, 1, 80, 100) - 1
y = torch.randn(N, 32) - 1

x_learn = x[:len(x)//2]
y_learn = y[:len(y)//2]

x_test = x[len(x)//2:]
y_test = y[len(y)//2:]

model.learn(x_learn, y_learn, EPOCHS, learning_rate=0.1, x_test=x_test, y_test=y_test)

print("VALIDATION : ", model.eval(x_test, y_test))

model.plot_losses()

# TODO Validation test in order to avoid overfitting
