import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( '-e','--epochs', help = 'Epochs to train the model', default = 2, type = int, required = False )
args = parser.parse_args( )

EPOCHS 		= args.epochs
CONV_FILTERS_NUM 	= 64
CONV_FILTER_SIZE 	= ( 3, 3 )
POOL_FILTER_SIZE 	= ( 2, 2 )

#####
#   #
# 1 # THE DATA
#   #
#####
# Load the data
( training_images, training_labels ), ( validation_images, validation_labels ) = datasets.cifar10.load_data( )

# Get data info
training_dataset_length, image_width, image_height, image_channels 	= training_images.shape
input_shape = ( image_width, image_height, image_channels )
validation_dataset_length 	= validation_images.shape[ 0 ]
classes_num = len( np.unique( training_labels ) )

# Process the data
training_images 	= training_images.reshape( 	training_dataset_length,	\
												image_width,				\
												image_height,				\
												image_channels )
training_images = training_images / 255.0
validation_images 	= validation_images.reshape( 	validation_dataset_length,		\
													image_width,					\
													image_height,					\
													image_channels )
validation_images = validation_images / 255.0


#####
#   #
# 2 # THE MODEL
#   #
#####

FIRST_LAYER 		= layers.Conv2D( 	CONV_FILTERS_NUM ,		\
										CONV_FILTER_SIZE, 			\
										activation = 'relu',		\
										input_shape = input_shape )
HIDDEN_LAYER_TYPE_1 = layers.MaxPooling2D( POOL_FILTER_SIZE )
HIDDEN_LAYER_TYPE_2 = layers.Conv2D( 	CONV_FILTERS_NUM,			\
										CONV_FILTER_SIZE,			\
										activation = 'relu' )
HIDDEN_LAYER_TYPE_3 = layers.MaxPooling2D( POOL_FILTER_SIZE )
HIDDEN_LAYER_TYPE_4 = layers.Conv2D( 	CONV_FILTERS_NUM, 				\
										CONV_FILTER_SIZE,				\
										activation = 'relu' )
HIDDEN_LAYER_TYPE_5 = layers.Dense( 64, activation = 'relu' )
LAST_LAYER 			= layers.Dense( classes_num, activation = 'softmax' )

model = models.Sequential( [
       FIRST_LAYER,
       HIDDEN_LAYER_TYPE_1,
       HIDDEN_LAYER_TYPE_2,
       HIDDEN_LAYER_TYPE_3,
       HIDDEN_LAYER_TYPE_4,
       layers.Flatten( ),
       HIDDEN_LAYER_TYPE_5,
       LAST_LAYER,
] )

#####
#   #
# 3 # Tune the model up.
#   #
#####
model.compile( optimizer = 'adam',								# Adam varies the learning rate (gradients) in order to
																# converge more quickly.
				  loss = 'sparse_categorical_crossentropy',		# Loss function useful for classification.
				  metrics = [ 'accuracy' ] )
model.summary()

#####
#   #
# 4 # Fit the model
#   #
#####
history = model.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS )

#####
#   #
# 5 # Evalueate the model.
#   #
#####
plt.plot( history.history[ 'accuracy' ] )
plt.plot( history.history[ 'val_accuracy' ] )
plt.title( 'model accuracy' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend( ['train', 'test'], loc = 'upper left' )
plt.xlim( [ 0, EPOCHS])
plt.ylim( [ 0.4 , 1.0 ] )
plt.show( )

#### end of file ####
