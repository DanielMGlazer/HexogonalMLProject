Hexagonal_Training_Data_Raw is the notebook used to create the training set. 
The _Raw refers to the switch from outputing .png style images to "raw" height values in array format. 
This could be dropped for all intents and purposes


The Defining functions of hexagonal grid and general helper functions section contains the methods that help construct a 2D grid of hexagons.
This was made following the instructions and guidelines of this guide: https://www.redblobgames.com/grids/hexagons/, and this is also where some of the weird termanology comes from.
q,r, and s are the coordinates for moving around on the hexagonal grid. Q stands for "q-ollumn" and r for row, which somewhat says how many rows or collums to move over from the origin.
Technically only q and r are required, but the guide also recomends using the s corrinate, as one can also visualize a hexagonal grid as all integer points in the 3D plane that fall on the 
plane x+y+z=0, or q+r+s=0. For most cases the s is ignored, and it can always be calculated when needed.


Following his guide one can construct the coordinate system for hexagons, which includes the locations of the corners of each hexagon. These are what we are really interested in for our project.
I only take the first 2 corners for every hexagon I construct, as otherwise there would be redundant corners when taking them from every point on the lattice. The whole process of making a 
hexagonal grid and finding the corners is complicated by strain. Strain is defined by 3 numers, q, r, and b. The letter b was more or less chosen at random. These correspond to strain pointing in
the q, r and "b" directions. Starting from the center of a flat top hexagon, q is facing the lower right face, r is facing the lower face, and b is facing the lower left face.
The numbers q, r, and b correspond to the percent strain in that direction, ie, how much those faces will be pushed apart from the center. A q value of .1 means the bond length in the q direction
will be 1.1 times the normal value. As the strain vectors point at faces and not corners, this isn't so straight forward as saying certain individual bonds will be 1.1 times as long. It may
have been easier to define strain in this matter, but what I currently have seems to work


Also in this section are rotation methods. These are self explanatory, of note is they all take a rotation matrix R. This is in the [[cos(),sin()],[-sin(),cos()]] format for some theta.
I also have some code that defines a few gaussian functions with arbitrary parameters. The ones I could find online were all normalized and I wanted to be sure I was using the right functions.
At the end there are methods to find the local minima and maxima of a given array





Class that creates training data:
This section creates the training data. 
The current list of parameters for making a training image are as follows:
bond_length: distance between two atoms in an unstrained lattice. Given as an array [x_length,y_length] as a leftover from the guide. Technically this can be used to strain in the x and y 
	direction, but I feed the same value for x and y and use the other parameters for strain.
atom_size: The atomic size, specifically the standard deviation in pixels of the gaussians used to plot each atom. As such the atoms will apear bigger than atom_size. Each gaussian is plotted
	out to 4*atom_size, a somewhat arbitrary choice.
offset: Some values [offset_x,offset_y] that shift the entire lattice in the x and y direction
strain: an array [q,r,b] of values defining strain in each of the three axes
angle: angle by which to rotate the whole lattice
corr: corrugation, height at which to draw each atom. Specifically this is the amplitude of the gaussians. Each atom aditionally gets a random height increase or decrease, so this is the 
	average height. This paramater should likely be removed, as the normalization done to all of the images prior to the model seeing them renders the variation in absolute height essentially
	worthless
smear_theta: angle by which to rotate the gaussian kernel used for smearing. 
Optional variations, useful for testing but usually one value is picked for the training set:
height_var: the standard deviation of the gaussian used to adjust the heights of each atom. ie if =0.05, most atom heights will be within 5% of corr
noise_percent: the multiplier in front of the scan line array. This array is first multiplied by corr to set it to the same average height as the atoms, and then by this to adjust the noise
b: the beta value in calculating the scan lines. While technicaly variable, we've never found the right value for our STM and have kept this at 1
N: The size of the scan line array from which we cut out a section. Should be set to the same size as whatever real STM images one is interested in. Currently set at 512

Without going in to the exact workings of the code, the general procedure is to make a rectangular map of hexagons a bit bigger than the intended image
as it is hard to exactly line the two up. Then one calculates all the atom locations, rotation and strain accounted for. Then all of the gaussians are drawn at these locations. Afterwards,
the gaussian smearing kernel is convolved over the image and scan lines are optionally added. The labeling process takes the list of all the atom locations and finds whichever atom is closest
to the numerical center of the image. From this atom location, the pre-perscribed neighbor vectors are added and the neighbor locations are found. The code does NOT search the remaining atom 
locations for the neighbors.




Studying Fourier Transform of simulated images:
Explained in Image Predictions README



Creation of Training Set:
This is the code that creates the training set. A gien size of training set is specified, and then arrays for each label are created.
Arrays for each parameter are also created here with the same number of entries as the training set size. The parameters all take random values, the current distributions are as follows:
bond length: truncated gaussian with mean=6, std=1. Truncated to values between 4.5 and 8
atom_size: truncated gaussian with mean=3.5, std=1. Truncated to values between 2.5 and 5.5
off_x/off_y: uniform distribution between -5 and 5
q/r/b: gaussian with mean=0, std=0.05
angle: uniform distribution between -60 and 60 degrees
corr: truncated gaussian with mean=0.15, std=0.05. Truncated to values between 0 and 0.4
smear_theta: uniform distribution between -90 and 90 degrees

These distributions were picked either to span the expected range of values or, in the case of bond length and atom size, to be close to what we expected in the real data for a 32x32 image. 
These could probably be studied with more care, I don't think they are currently the optimal values. They have also changed slightly throughout the project.

