import random
import numpy as np
import pickle



def getminibatches(Training_DataBase, Batch_size, N, savingpath) :	# take Batch_size = 32 and N = 128

	# N number of batches in a saving group
	Batches_Elemnts_Names = []	# initialize the list of batches elmnts names lists

	while len(Batches_Elemnts_Names) != L :	# split the whole Training_DataBase in Batches 
	
		for batch in range(N) :	# create a saving group of N batches



			### Initialization ###

			x_s = [0]*Batch_size
			x_w = [0]*Batch_size
			L_s = [0]*Batch_size 
			L_w = [0]*Batch_size
			L = len(Training_DataBase)

			### Filling of the batch matching elmnt ###

			while Training_DataBase[matching_Training_DataBase][2] in Batches_Elemnts_Names :	# make sure the elmnt is not already in a batch
				matching_Training_DataBase = random.randint(0,L-1)	# index of the matching elmnt in Training_DataBase

			matching_x_w = random.randint(0,L-1)	# index of the matching elmnt in x_w

			x_s[0] = Training_DataBase[matching_Training_DataBase][0]				# the matching pianoroll is the 1st elmnt of x_s
			x_w[matching_x_w] = Training_DataBase[matching_Training_DataBase][1]	# the matching STFT is a random index elmnt of x_w

			### Filling the matching elmnt name lists ###

			L_s[0] = Training_DataBase[matching_Training_DataBase][2]	
			L_w[matching_x_w] = Training_DataBase[matching_Training_DataBase][2]	

			### Filling all the rest to have a batch of size 32 ###

			x_w_index = [matching_x_w]	# initialize the list of x_w indices already filled
			Batch_Elmnt_Score_Names = [Training_DataBase[matching_x_w][2][:Training_DataBase[matching_x_w][2].rfind("_")]]	#Names in the batch regardless of the soundfont
			i = 0

			while i != Batch_size-1 :
				i += 1
				if len(Batches_Elemnts_Names) != L :	# make sure it remains elmnts to put in a batch 
					while Training_DataBase[matching_x_w][2][:Training_DataBase[matching_x_w][2].rfind("_")] in Batch_Elmnt_Score_Names :	# make sure we don't add an elmnt already in the batch regardless of the soundfont
						k = random.randint(0,L-1)

					x_s[i] = Training_DataBase[k][0]
					L_s[i] = Training_DataBase[k][2]
	
					while p in x_w_index :						# make sure we don't fill an index already filled in x_w
						p = random.randint(0,Batch_size-1)
	
					x_w[p] = Training_DataBase[k][1]
					L_w[p] = Training_DataBase[k][2]
					x_w_index.append(p)

			### Converting the lists x_s and x_w into np arrays ###

			x_s = np.asarray(x_s)	
			x_w = np.asarray(x_w)	

			### Adding the batch to others ###

			Batches = np.vhstack(Batches, np.hstack((x_s, x_w)))		# create a matrix of N batches in which a line is a batch
			Batches_Elemnts_Names.append([L_s, L_w])					# create a list of batches elmnts names lists

		## Group of batches "Batches" needs to be saved :  TO DO 

		#pickle.dump(, open( "save.p", "wb" ) )
