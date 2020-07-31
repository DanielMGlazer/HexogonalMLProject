Image Predictions is the notebook used for utilizing models to make predictions both on the training sets and the real data.



Predicting a dataset and error histograms:
This first section is made for generating predictions on the training set and comparing these to the training labels. While a machine learning purist may immediatley notice this 
goes against the principle of testing the model on the training data, I went for it anyway. In actuality, this was done because the training sets were being itterated on constantly,
the model's performance didn't change much when we did test it on seperately made testing sets, and most notibly the performance here wasn't the main sucess metric, as we were mostly
interested in a model's performance on real data.

The first cell loads the model into the model variable. NOTE: this is not advised if another notebook has a model saved in a variable, ie if one has trained a model recently and that kernel 
is still active. This caused my computer to lag heavily as having a single model loaded is taxing on RAM.

Next, the training set directory is specified, and a genertor is constructed that can pull data from that directory and feed it to the model. Important things to check here are that one's
directory is correct and that one has chosen the proper type of generator. This cell mimics the one in the model training code, however, these generators have the train variable set to False,
which tells the generator not to attatch labels to the data. The list of labels is still created in this cell for essentialy no purpose other than needing to feed the generator method something,
and it is likely an empty array could do the trick

Finally, the dataset is fed through the model in batches and the predictions are collected and saved in a pandas dataframe.

The subsequent cells take these predictions and plot histograms of the absolute difference between each prediction and its label. The get_error method creates an error array by simply subtracting
the prediction array from the label array. This requires both arrays to be ordered properly, but this usually hasn't been an issue and requires no extra steps at the moment. The plots are 
displayed below and optionally saved to a given directory. 

There is also a cell that overlays the histograms of two models. This is useful when comparing different model architectures or training sets.





Real STM Data Preprocessing:
This section is used to load in real stm images in .txt format. For the MoC2 images, no further work needed to be done, but for the graphene images, more preprocessing was needed
to subtract out any overarching shape. The flat graphene images, ie Graphene-SiC_159 or _160, required a planar subtraction, which is done in the first cell. The strained graphene had
a more complex underlying structure. Implemented currently is a fitting program for a sin wave in the x direction with a y dependent phase. It is likely a better fitting protocol would 
be nessecary to analyze this data further.


Applying a 2D Fourier Transform and mask cutouts
This seciton is for taking the Fourier transoform of the data, selecting some subregion of the FT image, and applying an inverse fourier transform to get a smoothed out image back.
There are various helper methods that make boolean masks of a certain shape. These can then be applied to the FT image. What's plotted isn't the direct fourier transform, it is the power spectrum
of the fourier transform. That being said, the masks take the same shape in both arrays and it is a useful visualization. The goal of this process is to both understand what type of noise
is present in the real STM data, as well as removing some portion of it so predictions can be made on cleaner data. Is is still to be decided how much noise can be removed without disturbing
the underlying atom locations or removing features of strain. The inverse fourier transformed image after the mask is applied is stored in the variable stm_back.

The next cell slices up both the original and fourier transformed version of the stm image. One can specify the size of the box that will be used to slice, as well as the step, the number of 
pixels between each slice. The original stm image is stored in stm_slice, whereas the fourier transformed image is stored in stm_slice_back. This naming convention is a leftover from some names
on the fourier transform code I took from whatever website. It might be prudent to change the word "back" to something more relevant. 

The last cell is a series of plotting methods as well as some leftover code for plotting images of these slices. Perhaps most useful here is the method plot_3d, which can be called on a 2D array
to quickly generate a 3D plot, however this only works if matplotlib is in "notebook" mode. To switch between the two modes, one calls either %matplotlib inline or %matplotlib notebook anywhere
in any cell block, and it will apply to the whole notebook. For me, I ran into the bug where the 3d plots would only show up if the %matplotlib inline command had not yet been run in this kernel
and one calls %matplotlib notebook for the first time. Switching back and forth seems to bug out the notebook style plots, although the inline plots do fine. Perhaps this is only an issue for me.
Left over in this cell is also some code that tracks your pointer and records its location at clicks. I used this to hand label one of the images, and it might be useful later. Feel free to 
move it somewhere else if it takes up too much space. 

After this there are cells defining the minima and maxima locators, as well as a cell to plot their results. 




Predicting on Real Data:
This section is for running the real stm data through the loaded model and plotting the results.

The first cell does this predicting. As the data we would like to give the model is in an array and not in a directory, we can not reuse the generators as they are to feed the data into the model.
As such, I recreate the steps the generator takes and feed each image in the array to the model in a single batch. This is more of a leftover from the time when I was only interested in single
32x32 images and not predicting the entire image at once. It is strongly advised that this section be tweaked such that the array is passed through the model in batches instead of one at a time.
For reference, the predictions on the training set (300k images) take around 2-3 minutes and each image must be loaded into ram. This process currently takes 30 minutes for a 256x256 array 
with a step of 1, and that only consists of 60k images already in ram. Regardless, these predictions are made and stored in an array generally called something related to the stm image in question.
A folder is also made for each model to store various predictions.

The next cells load in the prediciton array and split the data up into arrays containing the predicted points. This is not as simple as directly storing the information in the prediction array. As
the stm image is sliced up and resized into 32x32 images, these labels must be resized to whatever the original slicing box was. Additionally, the labels are given in distances relative to the
origin of each subimage, so these must be translated to wherever the slice was in the large stm image. This is all done by the get_split_labels_arr method. There is code for 8 labeled models as
well as 2 labeled models. 

The next cell is used to plot the predictions on top of the stm data. The methods present aid in plotting, as well as creating histograms of the data. There a few methods relating to averaging
nearest neighbors together; this was an earlier attempt to understand the data and may still be useful, although histograms are likely a better tool for analysis. The plots can be saved at the
end, by default going to prediction_arr_dir.

The final cell is used to make voronoi diagrams. The cell additionally finds the maxima in an image, although it does not use the above min/max finding methods.

