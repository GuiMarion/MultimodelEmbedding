
import torch
import torch.nn as nn
import torch.nn.functional as F

#Conv layer specs
in_channels = 1 # number of inputs in depth, 3 for a RGB image for example, 1 in our case
num_filters = 24 # Number of filters of the first conv layer cf article
kernel_size_conv = 3 # for a 3x3 convolution 

#Pool Layer specs, we want a downsample factor of 2 cf article
kernel_size_pool = 2
stride_pool = 2

dim_latent = 32 # Dimension of the embedding latent space cf article

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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


    def forward(self, x):

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
    	x = self.conv4_bn(self.conv4(x)) # Il manque l'opération linéaire Identity


    	#Global Pooling
    	global_pool = nn.AvgPool2d((x.size(2), x.size(3)))
    	x = global_pool(x)

    	#Flattening
    	x = x.view(dim_latent)

    	return x

# create a complete CNN
model = Net()
input = torch.randn(1,1,64, 64)

out = model(input)

print(out.size())
print(out)



