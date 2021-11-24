import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( '-e','--epochs', help = 'Epochs to train the model', default = 2, type = int, required = False )
args = parser.parse_args( )

EPOCHS = args.epochs

#################################################################################
#																			   	#
#	DNN
#																			   	#	
#################################################################################

#####
#   #
# 1 # LOAD DATA
#   #
#####
mnist = tf.keras.datasets.fashion_mnist
( training_images_orig, training_labels ), ( validation_images_orig, validation_labels ) = \
mnist.load_data( )

#####
#   #
# 2 # PROCESS DATA
#   #
#####
training_images 	= training_images_orig / 255.0
validation_images 	= validation_images_orig / 255.0

#####
#   #
# 3 # Create NN
#   #
#####
DNNmodel 	= tf.keras.models.Sequential( 									\
			[ tf.keras.layers.Flatten( ),									\
		  	  tf.keras.layers.Dense( 20, activation = tf.nn.relu ),			\
		  	  tf.keras.layers.Dense( 10, activation = tf.nn.softmax ) ] )

#####
#   #
# 4 # Tune the model up.
#   #
#####
DNNmodel.compile( 	optimizer = 'adam',							# Adam varies the learning rate (gradients) in order to
																# converge more quickly.
					loss = 'sparse_categorical_crossentropy',	# Loss function useful for classification.
					metrics = [ 'accuracy' ] )

#####
#   #
# 5 # Fit the model
#   #
#####
DNNmodel.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS )
DNNmodel.summary()

#####
#   #
# 6 # Evalueate the model.
#   #
#####
# This is already done during the fitting, but just for the sake of curiosity.
DNN_loss, DNN_accuracy = DNNmodel.evaluate(validation_images, validation_labels)

#################################################################################
#																			   	#
#	CNN
#																			   	#	
#################################################################################

#####
#   #
# 1 # LOAD DATA
#   #
#####
# Already done

#####
#   #
# 2 # PROCESS DATA
#   #
#####
# Now, THE FIRST CONVOLUTIONAL LAYER EXPECT EVERYTHING ALL TOGETHER, so we
# must reshape the whole list of 60000 images IN A SINGLE 4D TENSOR.
training_images=training_images_orig.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
validation_images=validation_images_orig.reshape(10000, 28, 28, 1)
validation_images=validation_images/255.0

# Get data info (for the sake of curiosity, not used at all)
dataset_length, image_width, image_height, image_channels 	= training_images.shape
print( "dataset_length:\t" + str( dataset_length ) )
print( "image_width:\t" + str( image_width ) )
print( "image_height:\t" + str( image_height ) )
print( "image_channels:\t" + str( image_channels ) )

#####
#   #
# 3 # Create NN
#   #
#####
CNNmodel	= tf.keras.models.Sequential( [
			# 2D CONVOLUTIONAL LAYER
			# param1(64)			-> number of filters.
			# param2( 3, 3 )		-> size of the filter.
			# param3( 'relu' )		-> activation method.
			# param4( 28, 28, 1)	-> shape of the input, 28x28 pixels by 1 channel (would be 3 if RGB)
			tf.keras.layers.Conv2D( 64, ( 3, 3), activation = 'relu', input_shape = ( 28, 28, 1 ) ),
			# MAX POOLING LAYER
			# param( 2, 2 ) 		-> size of the filter
			tf.keras.layers.MaxPooling2D( 2, 2 ),
			tf.keras.layers.Conv2D( 64, ( 3, 3), activation = 'relu' ),
			tf.keras.layers.MaxPooling2D( 2, 2 ),
			# Apply a flatten layer after all the filtering, from now on it is just a DNN
			tf.keras.layers.Flatten( ),
			tf.keras.layers.Dense( 20, activation = 'relu' ),
			tf.keras.layers.Dense( 10, activation = 'softmax' )
			] )

#####
#   #
# 4 # Tune the model up.
#   #
#####
CNNmodel.compile( optimizer = 'adam',								# Adam varies the learning rate (gradients) in order to
																	# converge more quickly.
				  loss = 'sparse_categorical_crossentropy',		# Loss function useful for classification.
				  metrics = [ 'accuracy' ] )
CNNmodel.summary()

#####
#   #
# 5 # Fit the model
#   #
#####
CNNmodel.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS )

#####
#   #
# 6 # Evalueate the model.
#   #
#####
# This is already done during the fitting, but just for the sake of curiosity.
CNN_loss, CNN_accuracy = CNNmodel.evaluate(validation_images, validation_labels)

#####
#   #
# 7 # DNN vs CNN Comparison.
#   #
#####

print( "\nDNN vs CNN Comparison:" )

print( "\nDNN Loss    : " + str( DNN_loss ) )
print( "DNN Accuracy: " + str( DNN_accuracy ) )

print( "\nCNN Loss    : " + str( CNN_loss ) )
print( "CNN Accuracy: " + str( CNN_accuracy ) + '\n' )

#### end of file ####
