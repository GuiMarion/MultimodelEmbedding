
class Modele():

	def __init__(self):

		self.model1 = None
		self.model2 = None

		self.X1 = []
		self.X2 = []
		self.L1 = []
		self.L2 = []
		
		self.x_test = None
		self.y_test = None
		
		self.losses = []
		self.losses_test = []


	def loadBatch(self, num):
		# Load mini batch from file named batch_num
        
		# upadte X1, X2, L1, L2


	def eval(self):
		# Copy and paste the eval function from network.py
		# You will surely have to implement a dummy loss function too
		

	def save_weights(self, name):
		# save the weights of the model with the name name
		self.model1.parameters().save("params/model1")
		self.model2.parameters().save("params/model2")
		

	def plot_losses(self):
        # plot the losses over time
        loss, = plt.plot(np.array(self.losses), label='Loss on training')
        lossTest, = plt.plot(np.array(self.losses_test), label='Loss on test')
        plt.legend(handles=[loss, lossTest])
        plt.show()
		

	def is_over_fitting(self):
		# return True of False is the modele is overfitting
		# find an algorithm that do the job i.e.
		# if self.losses_test is not decreasing for T epochs
		# with a threshold of K
        T = 10
		if(np.all(self.losses_test[-T:] < self.losses_test[-T]+K) == True):
            return True
        else:
            return False
    
	def learn(self, EPOCHS, learning_rate=1e-4, momentum=0.9):
		criterion = torch.nn.MSELoss(reduction='sum')
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = torch.optim.Adam(model1.parameters() + model2.parameters(), lr=learning_rate) ## if you can use +
        for t in range(EPOCHS):
        	# Make learn the two models with respects to x and y



			# append the losses to self.losses and self.losses_test
			self.losses.append(self.eval(self.X1, self.X2))
			self.losses_test.append(self.eval(self.x_test, self.y_test))
			
        	if self.is_over_fitting():
        		# stop learning
        		return





