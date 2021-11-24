'''
	In this script we will show how to get data ready within a "real" scenario.
	Steps to follow before applying the ML thing:
		1) Firstly download the kindly provided images. We have saved this on ../AI/datasets/
		$wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O horse-or-human.zip
		$wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O validation-horse-or-human.zip
'''

import argparse

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( '-e','--epochs', help = 'Epochs to train the model', default = 2, type = int, required = False )
args = parser.parse_args( )

EPOCHS 		= args.epochs

#####
#   #
# 1 # THE DATA
#   #
#####
import os
import zipfile
# Unzip the data.
datasets_path 	= '../../../datasets/'
datasets = [ 'horse-or-human', 'validation-horse-or-human' ]
for dataset in datasets:
	local_zip 		= datasets_path + dataset + '.zip'
	zip_ref 		= zipfile.ZipFile( local_zip, 'r' )
	output_path 	= datasets_path + dataset
	zip_ref.extractall( output_path )
zip_ref.close()

################################################################################
# Organize the data with ImageGenerator!!!!
################################################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CONFIGURE THE SINGLE IMAGE PROCESSING
# All images will be augmented according to whichever lines are uncommented below.
# we can first try without any of the augmentation beyond the rescaling
train_datagen = ImageDataGenerator(
      rescale 				= 1./255,
      #rotation_range 		= 40,
      #width_shift_range	= 0.2,
      #height_shift_range	= 0.2,
      #shear_range			= 0.2,
      #zoom_range			= 0.2,
      #horizontal_flip		= True,
      #fill_mode			= 'nearest'
      )

# CONFIGURE THE DATA FLOW
train_generator = train_datagen.flow_from_directory(
        datasets_path + 'horse-or-human/',  # This is the source directory for training images, the generator will
        									# automatically label each image with the directory name ("horses", "human")
        target_size = ( 100, 100 ),  		# All images will be resized to 100x100
        batch_size 	= 128,   				# Flow training images in batches of 128
        class_mode 	= 'binary')				# Since we use binary_crossentropy loss, we need binary labels

validation_datagen = ImageDataGenerator(
      rescale 				= 1./255,
      #rotation_range 		= 40,
      #width_shift_range	= 0.2,
      #height_shift_range	= 0.2,
      #shear_range			= 0.2,
      #zoom_range			= 0.2,
      #horizontal_flip		= True,
      #fill_mode			= 'nearest'
      )

validation_generator = validation_datagen.flow_from_directory(
        datasets_path + 'validation-horse-or-human',
        target_size 	= ( 100, 100 ),
        class_mode 		= 'binary')

################################################################################

#####
#   #
# 2 # THE MODEL
#   #
#####
import tensorflow as tf

# Design the NN
model = tf.keras.models.Sequential( [
    # Note the input shape is the desired size of the image with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D( 32, ( 3, 3 ), activation = 'relu', input_shape = ( 100, 100, 3 ) ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The second convolution
    tf.keras.layers.Conv2D( 64, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The third convolution
    tf.keras.layers.Conv2D( 128, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The fourth convolution
    tf.keras.layers.Conv2D( 256, ( 3, 3 ), activation = 'relu'),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten( ),
    # 512 neuron hidden layer
    tf.keras.layers.Dense( 512, activation = 'relu' ),
    tf.keras.layers.Dense( 256, activation = 'relu' ),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
] )
print( model.summary( ) )

# Select the optimizer
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop( learning_rate = 0.0001 )

# Build the guy.
model.compile(	loss 		= 'binary_crossentropy',
              	optimizer 	= optimizer,
              	metrics=[ 'acc' ] )

# Train the guy
history = model.fit(
      train_generator,
      steps_per_epoch 	= 8,
      epochs 			= EPOCHS,
      verbose 			= 1,
      validation_data 	= validation_generator )

#### end of file ####

#####
#   #
# 2 # USE THE MODEL
#   #
#####
from keras.preprocessing import image
import numpy as np

# Get image
validation_dataseth_path = datasets_path + 'validation-horse-or-human/'
images_path = [	validation_dataseth_path + '/horses/horse1-411.png',		\
				validation_dataseth_path + '/humans/valhuman01-00.png',		\
				validation_dataseth_path + '/horses/horse2-011.png',		\
				validation_dataseth_path + '/humans/valhuman02-06.png',		\
				validation_dataseth_path + '/horses/horse3-171.png',		\
				validation_dataseth_path + '/humans/valhuman03-09.png',		\
				validation_dataseth_path + '/horses/horse4-014.png',		\
				validation_dataseth_path + '/humans/valhuman04-11.png',		\
				validation_dataseth_path + '/horses/horse5-065.png',		\
				validation_dataseth_path + '/humans/valhuman05-21.png',		\
				validation_dataseth_path + '/horses/horse6-544.png' ]

print( "\nTESTING HOW GOOD THE MODEL PERFORMS:\n" )
for img_path in images_path:
	img = image.load_img( img_path, target_size = ( 100, 100 ) )

	# Transofrm to array.
	img_arr = image.img_to_array( img )
	img_arr = img_arr / 255.0
	img_arr = np.expand_dims( img_arr, axis = 0 )

	# Transform to tesnsor
	img_tensor = np.vstack( [ img_arr ] )

	classes = model.predict( img_tensor )
	if classes[ 0 ] > 0.5:
		if( "/humans/" in img_path ):
			print( "HUMAN OK" )
		else:
			print( "HORSE KO" )
	else:
		if( "/humans/" not in img_path ):
			print( "HORSE OK" )
		else:
			print( "HUMAN KO" )