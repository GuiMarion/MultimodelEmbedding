
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#Conv layer specs
in_channels = 1 # number of inputs in depth, 3 for a RGB image for example, 1 in our case
num_filters = 24 # Number of filters of the first conv layer 
kernel_size_conv = 3 # for a 3x3 convolution 

#Pool Layer specs, we want a downsample factor of 2
kernel_size_pool = 2
stride_pool = 2

dim_latent = 32 # Dimension of the embedding latent space 

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # History of the losses for cross-validation
        self.losses = np.array([])
        self.losses_test = np.array([])

        # convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size_conv, padding = 1)
        self.conv1_bn = nn.BatchNorm2d(num_filters)
        # convolutional layer 1bis
        self.conv1bis = nn.Conv2d(num_filters, num_filters, kernel_size_conv, padding = 1)
        self.conv1bis_bn = nn.BatchNorm2d(num_filters)

        # convolutional layer 2
        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size_conv, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(2 * num_filters)
        # convolutional layer 2bis
        self.conv2bis = nn.Conv2d(2 * num_filters, 2 *num_filters, kernel_size_conv, padding = 1)
        self.conv2bis_bn = nn.BatchNorm2d(2 * num_filters)

        # convolutional layer 3
        self.conv3 = nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size_conv, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(4 * num_filters)
        # convolutional layer 3bis
        self.conv3bis = nn.Conv2d(4 * num_filters, 4 * num_filters, kernel_size_conv, padding = 1)
        self.conv3bis_bn = nn.BatchNorm2d(4 * num_filters)

        # convolutional layer 5
        self.conv4 = nn.Conv2d(4 * num_filters, dim_latent, kernel_size = 1, padding = 0)
        self.conv4_bn = nn.BatchNorm2d(dim_latent)
    
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size_pool, stride_pool)

        self.dim_latent = dim_latent


    def forward(self, x):

        N = x.size()[0]

        # First Layer Conv2d + Elu + BN
        x = self.conv1_bn(F.elu(self.conv1(x)))
        # First bis Layer Conv2d + Elu + BN
        x = self.conv1bis_bn(F.elu(self.conv1bis(x)))
        # Max pool layer
        x = self.pool(x)

        # Second Layer Conv2d + Elu + BN
        x = self.conv2_bn(F.elu(self.conv2(x)))
        # Second bis Layer Conv2d + Elu + BN
        x = self.conv2bis_bn(F.elu(self.conv2bis(x)))
        # Max pool layer Conv2d + Elu + BN
        x = self.pool(x)

        # Third Layer Conv2d + Elu + BN
        x = self.conv3_bn(F.elu(self.conv3(x)))
        # Third bis Layer Conv2d + Elu + BN
        x = self.conv3bis_bn(F.elu(self.conv3bis(x)))
        # Max pool layer
        x = self.pool(x)

        # Fourth Layer Conv2d + Elu + BN
        x = self.conv3bis_bn(F.elu(self.conv3bis(x)))
        # Fourth bis Layer Conv2d + Elu + BN
        x = self.conv3bis_bn(F.elu(self.conv3bis(x)))
        # Max pool layer
        x = self.pool(x)

        #Fifth Layer Conv2d + Linear + BN
        x = self.conv4_bn(self.conv4(x))

        #Global Pooling
        global_pool = nn.AvgPool2d((x.size(2), x.size(3)))
        x = global_pool(x)

        #Flattening
        x = x.view(N, dim_latent)

        return x

    def learn(self, x, y, EPOCHS, learning_rate=1e-4, momentum=0.9, x_test=[], y_test=[]):
        criterion = torch.nn.MSELoss(reduction='sum')
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for t in range(EPOCHS):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.forward(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # append the losses to self.losses and self.losses_test
            self.losses = np.append(self.losses, self.eval(x,y))
            self.losses_test= np.append(self.losses_test, self.eval(x_test,y_test))
            
            if(t > 10 and self.is_over_fitting()):
                print("OVERFITTING!")
                break
            

    def loss_test(self, y_pred, y):
        # copute the loss for the final test part
        # use the MSE for now
        if len(y_pred) != dim_latent and len(y) != dim_latent:
            raise RuntimeError("y and y_pred dosn't have same shape for test.")

        loss = 0
        for i in range(dim_latent):
            loss += (float(y_pred[i]) - float(y[i]))**2

        return loss

    def eval(self, x, y):
        loss = 0
        y_pred = self.forward(x)
        for i in range(min(len(y_pred), len(y))):
            loss += self.loss_test(y_pred[i], y[i])

        loss /= min(len(x), len(y))

        return loss
        
    def plot_losses(self):
        # plot the losses over time
        loss, = plt.plot(np.array(self.losses), label='Loss on training')
        lossTest, = plt.plot(np.array(self.losses_test), label='Loss on test')
        plt.legend(handles=[loss, lossTest])
        plt.show()
        
    def save_weights(self, name):
        # save the weights of the model with the name name
        torch.save(self, "params/" + name + '.pt')
        
    def is_over_fitting(self):
        # return True is the modele is overfitting
        # find an algorithm that do the job i.e.
        # if self.losses_test is not decreasing for T epochs
        # with a threshold of K
        T = 10
        K = 0.1
        if(np.all(self.losses_test[-T:] > self.losses_test[-T]-K) == True):
            return True
        else:
            return False
