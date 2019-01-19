import os
from glob import glob
import pickle
from tqdm import tqdm
import random
import numpy as np
import torch

from Modules import score
from Modules import waveForm

# Parameters for the data extraction part
WINDOW_SIZE = 4 # in beat
STEP = 120 # in sample

TRAINSIZE = 0.6
TESTSIZE = 0.2
VALIDATIONSIZE = 0.2

DEBUG = False

FONTS = ["Full_Grand_Piano.sf2", "SteinwayGrandPiano_1.2.sf2", "FazioliGrandPiano .sf2"]



class dataBase:
    """Manages the database of music snippets ((piano roll, spectrogram, name) tuples).

    This class contains methods to build a database of (piano roll, spectrogram, name) tuples of music snippets
    from a repertory containing midi files, save this database in a file, or load a database from a file.
    It can also creates batches to feed the neural networks.

    Attributes
    ----------
    name : str
        Name of the database.
    data : :obj:`list`
        List of (piano roll, spectrogram, name) tuples for each snippet.
    path : str 
        Path of the midi database to load.
    """    
    
    def __init__(self, name="database"):

        self.name = name
        self.data = []
        self.path = None

    def constructDatabase(self, path, name=None):
        """Construct the database of tuples from an existing midi database.

        Parameters
        ----------
        path : str
            The path to the folder to load (must contain midi files).
        name : str, optional
            The name to give to the database object.
        """
        
        if os.path.isdir(path):
            self.path = path
            if name:
                self.name = name
            else:
                self.name = str(path)
            print()
            print("________ We are working on '" + path + "'")
            print()
        else:
            print("The path you gave is not a directory, please provide a correct directory.")
            raise RuntimeError("Invalid database directory")

        if not os.path.isdir(".TEMP"):
            os.makedirs(".TEMP")

        print("_____ Filling the database ...")
        print()

        # Number of skiped files
        skipedFiles = 0
        # Total number of files
        N = 0
        scores = []
        for filename in glob(self.path+'/**', recursive=True):

            if filename[filename.rfind("."):] in [".mid", ".midi"]:
                if os.path.isfile(filename):
                    print(" -", filename)
                    try : 
                        #ore_temp = score.score(filename)
                        #scores.extend(score_temp.extractAllParts(WINDOW_SIZE, step=STEP))
                        scores.append(score.score(filename))

                    except RuntimeError:
                        skipedFiles += 1
                    N += 1

        print()
        print("We passed a total of ", N, "files.")
        print(skipedFiles,"of them have been skiped.")
        print()

        print("_____ Augmenting database ...")
        print()

        #scores = self.augmentData(scores)

        print("_____ Computing the sound ...")
        print()

        shapes1 = []
        shapes2 = []

        for s in tqdm(scores):
            waveforms = []
            spectrum_temp = []

            N = s.length * s.quantization
            windowSize = WINDOW_SIZE * s.quantization
            retParts = []
            r = []

            for f in range(len(FONTS)):
                spectrum_temp.append(s.toWaveForm(font=FONTS[f]).getCQT())
                r.append(spectrum_temp[f].shape[1] / s.getPianoRoll().shape[1])


            tmpPart1 = []
            tmpPart2 = []
            for i in range((N-windowSize)//STEP):
                tmpPart1 = s.extractPart(i*STEP, i*STEP+windowSize)
                tmpPart2 = []
                tmpNames = []
                for f in range(len(FONTS)):

                    tmpPart2.append(spectrum_temp[f][:,round(i*STEP*r[f]) : round(i*STEP*r[f]) + round(windowSize*r[f])])
                    tmpNames.append(tmpPart1.name + "-" + FONTS[f])

                self.data.append( (tmpPart1.getPianoRoll(), tmpPart2, tmpNames) )

                if DEBUG:
                    if str(tmpPart1.getPianoRoll().shape) not in shapes1:
                        shapes1.append(str(tmpPart1.getPianoRoll().shape))
                    if str(tmpPart2[0].shape) not in shapes2:
                        shapes2.append(str(tmpPart2[0].shape))

        random.shuffle(self.data)

        if DEBUG :
            print("Shape 1")
            print(shapes1)
            print()
            print("Shape 2")
            print(shapes2)

    def save(self, path="../DataBase/Serialized/"):
        """Saves the database as a pickle.

        Parameters
        ----------
        path : str, optional
            The path to the folder in which we save the file.
        """

        answer = "y"

        if os.path.isfile(path+self.name+'.data'):
            print(path + self.name + ".data" + " " + " already exists, do you want to replace it ? (Y/n)")
            answer = input()

            while answer not in ["", "y", "n"]:
                print("We didn't understand, please type enter, 'y' or 'n'")
                answer = input()

            if answer in ["", "y"]:
                os.remove(path+self.name + '.data')

        if answer in ["", "y"]:
            print("____ Saving database ...")
            f = open(path+self.name + '.data', 'wb') 
            pickle.dump(self.data, f)
            f.close()

            print()
            print("The new database is saved.")
        else:
            print()
            print("We kept the old file.")


    def load(self, path):
        """Loads  a database from a previously saved pickle.

        Parameters
        ----------
        path : str
            The path to the folder containing the previously saved database.
        """
        
        if not os.path.isfile(path):
            print("The path you entered doesn't point to a file ...")
            raise RuntimeError("Invalid file path")

        try:
            self.data = pickle.load(open(path, 'rb'))
            print("We successfully loaded the database.")
            print()
        except (RuntimeError, UnicodeDecodeError) as error:
            print("The file you provided is not valid ...")
            raise RuntimeError("Invalid file")

    def print(self):
        """Prints name of all items in the database."""
        
        print("____Printing database")
        print()
        for i in range(len(self.data)):
            print(self.data[i][2])

    def getData(self):
        """Returns the content of the database."""        
        
        return self.data

    def augmentData(self, scores):
        """Augments the data with some techniques like transposition.
        
        Parameters
        ----------
        scores : :obj:'list'
            List of scores to augment.

        Returns
        -------
        augmentedData : :obj: 'list'
            List containing the scores resulting from the augmentation.
        """
        
        augmentedData = []

        for s in tqdm(scores):
            augmentedData.extend(s.getTransposed())


        return augmentedData


    def getBatches(self, batchSize):
        """Returns efficiently valid batches from data if len(data) > 32.
        
        Parameters
        ----------
        batchSize : int
            Size of the batches to create.        
        
        Returns
        -------
        batches : :obj: 'list'
            List containing the resulting batches.
        """

        batches = []
        for i in tqdm(range(int(TRAINSIZE * len(self.data)) // batchSize)):
            batches.extend(self.getSmallSetofBatch(self.data[i*batchSize : (i+1)*batchSize], batchSize))

        return batches

    def getSmallSetofBatch(self, data, batchSize):
        """Constructs valid batches from data, is efficient if len(data) <= 32.
        
        Parameters
        ----------
        data : :obj: 'list'
            Content of the database.        
        batchSize : int
            Size of the batches to create.

        Returns
        -------
        batches : :obj: 'list'
            List containing the resulting batches.
        """

        numberofData = len(data)*len(data[0][2])
        numberofBatches = numberofData // batchSize
        batches = []
        empty = []

        for b in range(numberofBatches):

            X1 = []
            X2 = []
            L1 = []
            L2 = []
            unseenX = np.delete(np.arange(len(data)), empty)
            for i in range(batchSize):
                k1 = np.random.randint(len(unseenX))
                k = unseenX[k1]
                X1.append(data[k][0])
                k2 = np.random.randint(len(data[k][1]))
                X2.append(data[k][1][k2])
                L1.append(data[k][2][0][:data[k][2][0].find("-")])
                L2.append(data[k][2][k2])
                # We delete the indice we just saw
                unseenX = np.concatenate((unseenX[:k1], unseenX[k1+1:]), axis=None)
                # We also delete the spectrum we used
                del data[k][1][k2]
                del data[k][2][k2]

                if len(data[k][2]) == 0 :
                    empty.append(k)

            # We shuffle X2 in order to don't the make matching indices in each batch
            # We need to shuffle L2 the same way in order to get the right names
            indices = np.arange(len(L1))
            c = list(zip(X2, L2, indices))
            random.shuffle(c)
            X2, L2, indices = zip(*c)
            
            batches.append((X1, X2, L1, L2, indices))


        return batches

    def getTestSet(self, batchSize):
        """Constructs batches for the test set.

        Parameters
        ----------    
        batchSize : int
            Size of the batches to create.

        Returns
        -------
        batches : :obj: 'list'
            List containing the resulting batches.
        """ 

        batches = []
        for i in tqdm(range(int(TRAINSIZE * len(self.data)) // batchSize, int((TRAINSIZE + TESTSIZE) * len(self.data)) // batchSize)):
            batches.extend(self.getSmallSetofBatch(self.data[i*batchSize : (i+1)*batchSize], batchSize))

        return batches

    def getValidationSet(self, batchSize):
        """Constructs batches for the validation set.

        Parameters
        ----------    
        batchSize : int
            Size of the batches to create.

        Returns
        -------
        batches : :obj: 'list'
            List containing the resulting batches.
        """ 

        batches = []
        for i in tqdm(range(int((TRAINSIZE + TESTSIZE) * len(self.data)) // batchSize, len(self.data) // batchSize)):
            batches.extend(self.getSmallSetofBatch(self.data[i*batchSize : (i+1)*batchSize], batchSize))

        return batches

