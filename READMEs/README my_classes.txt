my_classes is a python script containing custom classes I use in other parts of the code. Most notably this contains the generators used to feed data to the model

A generator is an object that can infinitely supply batches of data and labels upon request from some directory. In our case, we supply the generators with a list of filenames
and a dictionary linking those filenames to their labels. I will go through the structure of DataGeneratorAug, as it is the more fundamental case, but the other generators are 
very similar with minor tweaks. All of these designs are based off of this guide: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly  



DataGenerator Aug looks like this:
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
        y = np.empty((self.batch_size,2)) #Empty array to fill with labels. Second value must be the number of labels for this model

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data=np.load(os.path.join(self.directory,ID))
            data=data_aug(data,self.norm,self.dim)
            X[i,] = data

            # Store class
            y[i] = self.labels[ID]

        return X, y




The __init__ method assigns the input variables to instance variables. 
	dim is the dimension the input image should be transformed too, usually the same size in the first
		two dimensions and 1 in the 3rd dimension. 
	Batch_size is the batch size. 
	labels is the dictionary of labels
	list_IDs is the list of ID's, or filenames
	directory is the filepath towards where the training data is stored
	shuffle is a boolean for whether or not to shuffle the indicies of the array before each batch. This ensure each batch has a random sample from the training
		set. Not strictly nessesary since our training set is already random, but good practice overall
	train is a boolean for whether this generator is for training or predicting. If train=False, labels will not be assigned to the data and the generator will return
		only augmented data
	norm is the type of normalization scheme used, in this case either "standardize" or "divmax"

The __getitem__ method generates one batch of data. It takes as input an index and will return a batch from that index to index+batch size from the ID list. For example,
if you give it index 5 and batch size 200, it will return lattice_1000 through lattice_1200 if the indicies aren't shuffled. The model knows that if it's fed a generator, 
it should call this method a given number of times with an increaseing index. How many times the model calls this method per epoch is given to the model.fit method. Once the 
method has its list of indicies for this batch, it gets the corresponding list ID's, and then calls the __data_generation method to actually make the batch. This batch is then
returned, with or without labels depending on the train variable.

The on_epoch_end method resets the indicies and optionally shuffles them

The __data_generation method takes a list of ID's as input and generates a batch. For each ID it recieves, the method loads that file from the directory, performs whatever data
augmentation is requested on it, and stores it in the batch array called X. Additionally, as the labels are stored in a dictionary, every ID is paired up with its label in order
and the labels are stored in the batch array Y. The data augmentation method is defined ouside the class. This batch is then returned.

The data augmentation method is here:

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
       
    data=np.reshape(data,dim)
    return data  

It performs a normalization of the data, then reshapes it to the given dimension. "standardize" means to subtract the mean value and divide by the standard deviation, "divmax"
means to subtract the mean and divide by the maximum value. Both of these reduce the values to some standardized range, but divmax ensures this range is between -1 and 1.

