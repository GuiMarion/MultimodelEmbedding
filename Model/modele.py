import numpy as np
import network
import torch
import pickle


class Modele():

        def __init__(self, databasePath=None):

                self.model1 = network.Net()
                self.model2 = network.Net() 

                self.batch_size = 32

                self.databasePath = databasePath

                self.loadBatch(0)


                self.losses = []
                self.losses_test = []


        def loadBatch(self, num):
                # Load mini batch from file named batch_num

                print("Loading batch: ", num)

                if self.databasePath is None:

                        self.X1 = [torch.randn(self.batch_size, 1, 80, 100)-1 for i in range(self.batch_size)]
                        self.X2 = [torch.randn(self.batch_size, 1, 80, 100)-1 for i in range(self.batch_size)]
                        self.L1 = [str(i) for i in range(self.batch_size)]
                        self.L2 = [str(j) for j in range(self.batch_size)]

                else:
                        self.X1 = []
                        self.X2 = []
                        self.L1 = []
                        self.L2 = []

                        data = pickle.load(open(self.databasePath, 'rb'))
                        for i in range(len(data)):
                                self.X1.append(data[i][0])
                                self.X2.append(data[i][1][:,:39]) # just for the moment tfct has to be refactored
                                self.L2.append(data[i][2][:data[i][2].rfind("-")])
                                self.L2.append(data[i][2])

                        self.X1 = np.stack(self.X1, axis=0)
                        self.X2 = np.stack(self.X2, axis=0)
                        
                        self.X1 = torch.from_numpy(self.X1.reshape(len(data), 1, self.X1.shape[1], self.X1.shape[2]).astype(float)).float()
                        self.X2 = torch.from_numpy(self.X2.reshape(len(data), 1, self.X2.shape[1], self.X2.shape[2]).astype(float)).float()

                        temp1 = []
                        temp2 = []

                        for i in range(len(data) // self.batch_size):
                                temp1.append(self.X1[i*self.batch_size : (i+1)*self.batch_size])
                                temp2.append(self.X2[i*self.batch_size : (i+1)*self.batch_size])

                        self.X1 = temp1
                        self.X2 = temp2

        def loss_test(self, y_pred1, y_pred2):
                # copute the loss for the final test part
                # use the MSE for now
                if len(y_pred1) != dim_latent and len(y_pred2) != dim_latent:
                        raise RuntimeError("y_pred1 and y_pred2 doesn't have same shape for test.")

                loss = 0
                for i in range(dim_latent):
                        loss += (float(y_pred1[i]) - float(y_pred2[i]))**2

                return loss


        def eval(self, X1, X2):
                # Copy and paste the eval function from network.py
                # You will surely have to implement a dummy loss function too

                loss = 0

                y_pred1 = self.forward(X1)
                y_pred2 = self.forward(X2)

                for i in range(min(len(y_pred1), len(y_pred2))):
                        loss += self.loss_test(y_pred1[i], y_pred2[i])

                loss /= min(len(X1), len(X2))

        def save_weights(self, name):
                # save the weights of the model with the name name

                pass


        def plot_losses(self):
                # plot the losses over time

                pass

        def is_over_fitting(self):
                # return True of False is the modele is overfitting
                # find an algorithm that do the job i.e. 

                # if self.losses_test is not inscreasing for T epochs
                # with a threshold of K

                return False

        def learn(self, EPOCHS, learning_rate=1e-7, momentum=0.9):
                criterion = torch.nn.MSELoss(reduction='sum')
                #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
                parameters = [p for p in self.model1.parameters()] + [p for p in self.model2.parameters()]

                optimizer = torch.optim.Adam(parameters, lr=learning_rate) ## if you can use +
                for t in range(EPOCHS):
                        # Make learn the two models with respects to x and y

                        y_pred_test1 = self.model1.forward(self.X1[t % self.batch_size])
                        y_pred_test2 = self.model2.forward(self.X2[t % self.batch_size])

                        # Compute and print loss
                        loss = criterion(y_pred_test1, y_pred_test2)
                        print(t, loss.item())

                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if t % self.batch_size == self.batch_size -1:
                                self.loadBatch(t // self.batch_size +1)


                        # append the losses to self.losses and self.losses_test

                        if self.is_over_fitting():
                                # stop learning
                                return






