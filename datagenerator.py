import numpy as np
from numpy import newaxis
from tensorflow.keras.utils import Sequence
import os

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,directory=None, batch_size=32, dim=(32,32), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.directory= directory
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size,8))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #x=np.load(os.path.join(self.directory,ID))
            X[i,] = np.reshape(np.load(os.path.join(self.directory,ID)),(self.dim[0],self.dim[1],1))

            # Store class
            y[i] = self.labels[ID]

        return X, y