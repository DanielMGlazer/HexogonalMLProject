#Packages
import numpy as np
from numpy import newaxis
from tensorflow.keras.utils import Sequence
import os
import collections
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor
from scipy.ndimage.filters import gaussian_filter

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,directory=None, batch_size=32, dim=(32,32,1), shuffle=True, train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.length=len(list_IDs)
        self.directory= directory
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size<=self.length:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes=self.indexes[index*self.batch_size:self.length]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.train==True:
            return X, y
        
        else:
            return X

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
    
def data_aug(data, norm,dim):#Performs data aumentation before being fed into batch
    assert norm == 'standardize' or norm == 'divmax'
    #Standardization
    if norm == 'standardize':
        data -= np.mean(data)
        data /= np.std(data)

    #Divide out the max
    if norm == 'divmax':
        data -= np.mean(data)
        data_max=np.max(data)
        data /= data_max
       
    #data=data[0:32,0:32]
    data=np.reshape(data,dim)
    return data   
    

def label_aug(label):
    new_label=np.empty((8))
    new_label[0]=label[0]
    new_label[1]=label[1]
    new_label[2]=label[0]+label[2]
    new_label[3]=label[1]+label[3]
    new_label[4]=label[0]+label[4]
    new_label[5]=label[1]+label[5]
    new_label[6]=label[0]+label[6]
    new_label[7]=label[1]+label[7]
    return new_label
    
    
class DataGeneratorAug(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=[0],directory=None, batch_size=32, dim=(32,32,1), shuffle=True, train=True,
                norm='standardize'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.length=len(list_IDs)
        self.directory= directory
        self.shuffle = shuffle
        self.train = train
        self.norm=norm
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size<=self.length:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes=self.indexes[index*self.batch_size:self.length]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.train==True:
            return X, y
        
        else:
            return X

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
            data=np.load(os.path.join(self.directory,ID))
            data=data_aug(data,self.norm,self.dim)
            X[i,] = data

            # Store class
            y[i] = self.labels[ID]

        return X, y
    
     
class DataGeneratorMultiInputAug(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, extra_inputs, directory=None, batch_size=32, dim=(32,32,1), shuffle=True, train=True,
                 norm='standardize'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.extra_inputs = extra_inputs
        self.list_IDs = list_IDs
        self.length = len(list_IDs)
        self.directory = directory
        self.shuffle = shuffle
        self.train = train
        self.norm = norm
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size<=self.length:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes=self.indexes[index*self.batch_size:self.length]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [X,X1], y = self.__data_generation(list_IDs_temp)

        if self.train==True:
            return [X,X1], y
        
        else:
            return [X,X1]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        X1= np.empty((self.batch_size,1))
        y = np.empty((self.batch_size,8))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data=np.load(os.path.join(self.directory,ID))
            data=data_aug(data,self.norm,self.dim)
            X[i,] = data
            X1[i] = self.extra_inputs[ID]

            # Store class
            if self.train==True:
#                 label=self.labels[ID]
#                 label=label_aug(label)
                y[i] = self.labels[ID]

        return [X,X1], y

class STMImage:

    background_color=ImageColor.getrgb("hsl(46,0%,50%)")
    def __init__(self,height_arr,image_size=[32,32]):
        self.image_size=image_size
        self.im=Image.new("RGB",self.image_size,color=self.background_color)
        self.pixel_array=np.array(self.im)
        self.blob_values_array=height_arr
        self.make_im()
        

    #Method for making 2D colored image
    #Takes all float values in blob_values_array and converts them to an hsl color with the corresponding lightness value. This color is then convereted to rgb and stored in pixel_array   
    def gauss_to_color(self):
        max_val=self.blob_values_array.max()
        self.rescaled_heights=self.blob_values_array*100/max_val
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                rgbcolor=ImageColor.getrgb(f"hsl(46,{min(max(self.rescaled_heights[j][i],1e-2),100)}%,50%)")
                self.pixel_array[j][i][0]=rgbcolor[0]
                self.pixel_array[j][i][1]=rgbcolor[1]
                self.pixel_array[j][i][2]=rgbcolor[2]


   
        
    #Runs the methods required to draw the image and saves the final array as an Image instance
    def make_im(self):
        self.gauss_to_color()
        self.image=Image.fromarray(self.pixel_array)
  