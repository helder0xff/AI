import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2
from keras.preprocessing import image

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( 	'-e',
						'--epochs',
						 help = 'Epochs to train the model', 
						 default = 2, 
						 type = int, 
						 required = False )
parser.add_argument( 	'-sm',
						'--streammode',
						help = 'Stream Mode', 
						default = 'array', 
						type = str, 
						choices = ['array', 'image'], 
						required = False )
parser.add_argument( 	'-c',
						'--channels',
						help = 'RGB channels. Choose when using streammode = array.', 
						default = 'RGB', 
						type = str, 
						choices = ['R', 'G', 'B', 'RG', 'RB', 'GB', 'RGB'], 
						required = False )
args = parser.parse_args( )

EPOCHS      = args.epochs
STREAMMODE 	= args.streammode
if 'image' == STREAMMODE:
	CHANNELS = 'original'
else:
	CHANNELS 	= args.channels

# SOME CONFIGURATION
BATCH_SIZE 	= 128
OPTIMIZER = 'adam'
if 'image' == STREAMMODE:
	LOSS_FUNCTION = 'categorical_crossentropy'
else:
	LOSS_FUNCTION = 'sparse_categorical_crossentropy'
# Harcoded for original image training. Got from the 04a_GetDataReady.py script
# This is the smallest image resolution.
SHAPE = (240,143,3)	

#####
#	#
# 1 # THE DATA
#	#
#####
flowers = [ '/daisy', '/dandelion', '/roses', '/sunflowers', '/tulips' ]
datasets_path 		= '../../../datasets/flower_photos'
channels 			= '/' + CHANNELS
training_path		= datasets_path + '/training' + channels
validation_path 	= datasets_path + '/validation' + channels
test_path 			= datasets_path + '/test' + channels

# GET THE SHAPE OF THE DATASET
if 'array' == STREAMMODE:
	npy = np.load( training_path + flowers[0] + '/1.npy' )
	SHAPE = npy.shape

# STREAM MODE
# The data flow from the dataset is different depending on if the images
# are saved as regular format (.jpg, .png, ...) or in the form of merged
# channels numpy arrays (.npy)

# When we have .jpg images, TF make the labelling magic itself...
if 'image' == STREAMMODE:
	training_datagen = ImageDataGenerator(
		rescale = 1./255 )
	training_generator = training_datagen.flow_from_directory(
		training_path,
		target_size = ( SHAPE[0], SHAPE[1] ),
		batch_size 	= (BATCH_SIZE),
		class_mode 	= 'categorical' )

	validation_datagen = ImageDataGenerator(
		rescale = 1./255 )
	validation_generator = validation_datagen.flow_from_directory(
		validation_path,
		target_size = (SHAPE[0], SHAPE[1]),
		batch_size 	= BATCH_SIZE,
		class_mode 	= 'categorical' )
#... otherwise we manually do it.
else:
	training_dataset 	= []
	training_labels		= []
	index = 0
	for flower in flowers:
		path = training_path + '/' + flower + '/'		
		for file in os.listdir( path ):	
			npy = np.load( path + file )
			training_dataset.append( npy / 255.0 )
			training_labels.append( index )
		index += 1
	training_dataset 	= np.array(training_dataset)
	training_labels 	= np.array(training_labels)

	validation_dataset 	= []
	validation_labels	= []
	index = 0
	for flower in flowers:
		path = validation_path + '/' + flower + '/'
		for file in os.listdir( path ):
			npy = np.load( path + file )
			validation_dataset.append( npy / 255.0 )
			validation_labels.append( index )
		index += 1
	validation_dataset 	= np.array(validation_dataset)			
	validation_labels 	= np.array(validation_labels)

#####
#	#
# 2 # THE MODEL
#	#
#####
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image with 3 bytes color
    # This is the first convolution    
    tf.keras.layers.Conv2D( 16, ( 3, 3 ), activation = 'relu', input_shape = SHAPE ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The second convolution
    tf.keras.layers.Conv2D( 32, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The third convolution
    tf.keras.layers.Conv2D( 64, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The fourth convolution
    tf.keras.layers.Conv2D( 128, ( 3, 3 ), activation = 'relu'),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten( ),
    # 512 neuron hidden layer
    tf.keras.layers.Dense( 512, activation = 'relu' ),
    # 5 flowers -> 5 neurons
    tf.keras.layers.Dense( 5, activation = 'softmax' )
])

# Some NN info.
model.summary()

model.compile(
    loss = LOSS_FUNCTION,
    optimizer = OPTIMIZER,
    metrics = ['accuracy']
)

if 'image' == STREAMMODE:
	history = model.fit(
	      training_generator, 
	      epochs = EPOCHS,
	      verbose = 1,
	      validation_data = validation_generator)
else:
	history = model.fit(
		training_dataset,
		training_labels,
		epochs = EPOCHS,
		validation_data = (validation_dataset, validation_labels)
		)
#####
#	#
# 2 # TEST
#	#
#####
# Just check if our testing dataset is properly predicted.
print( "\nTesting...\n" )
numberTest = 0
numberSuccess = 0
for flower in flowers:
	path = test_path + '/' + flower + '/'
	for file in os.listdir( path ):
		numberTest += 1
		if 'image' == STREAMMODE:
			test = image.load_img( path + file, target_size = ( SHAPE[0], SHAPE[1] ) )
			test = image.img_to_array( test )
			test = test / 255.0
			test = np.expand_dims( test, axis = 0 )	
			test = np.vstack([test])	
		else:
			test = np.load( path + file )
			test = np.array(test/255)
			test = tf.expand_dims(test, axis=0)
		
		#img = image.load_img( path + file, target_size = TARGET_SIZE )

		#img_arr = image.img_to_array( img )
		#img_arr = img_arr / 255.0
		#img_arr = np.expand_dims( img_arr, axis = 0 )

		#img_tensor = np.vstack( [ img_arr ] )
		
		classes = model.predict( test )
		
		if( flower == flowers[ np.argmax(classes) ] ):
			numberSuccess += 1
			print( "RIGTH", end='\r' )
		else:
			print( "WRONG", end='\r' )

print("#######################################################################")
print( CHANNELS, "accuracy:", str( round( numberSuccess * 100 / numberTest, 2 ) ) + '%' )
print( 'Images resolution:', SHAPE)
print("#######################################################################")

#### end of file ####
