Hexagonal Model Training 

This is the notebook responsible for creating and training new models, as well as plotting the model histories.

The first cell block imports the dataframe of choice. This is accomplished by setting the proper directory that contains the dataframe. In my case this was also the directory that contained
the training sets. The dataframe is stored in the global variable df for the rest of the notebook.

The next block is for constructing the model. This can either be done with the Keras Sequential class, which makes things a bit easier and more streamlined, or through the keras functional api,
a more customizable way to construct models. For linear models with one input, the sequential class is recommended. 
The models are constructed by adding given layers in order and then calling the compile method. The most recent model architecture is as follows:
def build_model():
    model = models.Sequential()                                 
    model.add(layers.Conv2D(32, (3,3) ,activation='relu',padding='same', input_shape=(32,32,1)))
    model.add(layers.MaxPooling2D((2,2),padding="same"))
    model.add(layers.Conv2D(64,(3,3), activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2),padding='same'))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2))
    return model

def get_compiled_model():
    model=build_model()
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=['mse','mae'])
    return model

The model starts by creating a model instance from the Sequential class. From here layers can be added. The first layer is a 2D convolutional layer, with the number of nodes and the kernel
size specified, here with 32 nodes and a 3x3 kernel. In each layer, one must specify the activation funciton; I use relu for all my layers. For the first layer, the sequential class assumes
this is the input layer, so an input shape must be specified. Here (32,32,1) means a 32x32 image with one chanel. Noteably this is the same amount of data as a (32,32) numpy array, indeed 
that's what all my images start as, however the Conv2D method is built for handeling images with channels, and these are 3D data formats. As such it is nessesary to reshape even 2D arrays into
these 3d formats for the model to recognize the data. On the first two convolutional layers I have padding='same', this indicates that the convolutional kernel should apply padding to 
the edges of the data as it scans across. This adds extra borders of zeros and means the output will have the same size as the input. Following the first conv layer is a MaxPooling layer. 
This layer has a kernel of (2,2), as it is recomended to use a max pooling kernel one unit smaller than the convolutional kernel. I have three convolutional/pooling layers, followed by a flatten
layer which flattens the output of the last layer into a 1d array the dense layers can recognize. I then have three dense layers, whose only parameters are the number of nodes and the activation 
function. In between them I have dropout layers, which randomly block a percentage of the connections during training. This prevents any one connection from growing too strong/important, and helps
the model with overfitting. The final dense layer has no activation function, as this is the output layer. As we are doing regression, we want whatever physical value the last layer takes,
so no additional function is needed to interpret this result. The model instance is then returned, containing all of the specified layers.

The compile method constructs a model with the build_model method, then complies it with specified optimizer, loss function, and metrics. I use Adam for the optimizer, which is a popular
optimizer for image recognition taks. I use root mean squared as my loss function, which is a popular loss metric for regression tasks. It is the square root of the sum of the squares 
of all the errors between labels and predictions in a batch. Thus, the loss is on the order of what we are trying to measure, but it still magnifies small errors. That being said, using
different loss functions could be a good area of study just to confirm this is a good choice. As root mean squared is not a prebuilt loss function in keras (for whatever reason, it is a prebuilt
metric) I had to code in my own version, which is defined in the compile method as well. Lastly are the metrics, which are kept track of like the loss funciton but are not used in training.
These can also be valuable information about the model. I used mean squared error and mean absolute error.



The next cell is similar, however this constructs a model using the keras functional API. The main difference is instead of adding layers in sequence to a model instance, one makes a bunch of 
different layer instances where each layers has a defined input. This way you can connect all the different layers any way you want. This construction is useful for having layers with 
more than one input, having inputs come into the model layer, or making models with multiple outputs. Branching structures and the like require the functional API. Visit 
https://keras.io/guides/functional_api/ for more information. 


The next cell creates the training and validation sets, as well as constructing the genertors to be used in training the model. 
There are parameters to control the validation split and the batch size. I've used 0.2 and 200. I think these are pretty reasonable numbers and likely don't need to be changed. 
The filenames from the original dataframe are then put into arrays for the validation and training sets, and then the labels are put into dictionaries such that each label is paried with the
right image. These dictionaries are fed into the generators to be used in training. 
The generators are better described in the my_classes readme, however in summary, they are a way to feed the data into the model without loading the entire training set into RAM. They also
provide a way to do image preprocessing on the fly for things that are applied to every image in the training set. The standard generator is DataGeneratorAug, which expects a single image 
as input to the model. DataGeneratorMultiInputAug expects an image and a float, we used this to include a guess for the bond length. DataGeneratorPeakMask expects an image and a mask of the same
size, inputting an array of (32,32,2) to the model. We used this to input a boolean mask of the local minima or maxima. 



The next cell actually trains the model. The number of epochs is set by num_epochs. The callbacks used are EarlyStopping and ModelCheckpoint. This stops training if the validation loss stops 
improving for a given number of epochs, and ModelCheckpoint saves the model at given checkpoints in case training stops for whatever reason before reaching the end. The model is saved in 
its own folder, given by model_name.


The next cell plots the loss and other metrics tracked for the model, saving them to a folder also labeled by the model name under "History Plots".

