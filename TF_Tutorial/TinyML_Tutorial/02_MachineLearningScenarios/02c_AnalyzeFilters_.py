import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( '-e','--epochs', help = 'Epochs to train the model', default = None, type = int, required = True )
args = parser.parse_args( )

EPOCHS = args.epochs

#####
#   #
# 1 # LOAD DATA
#   #
#####
# Load data.
mnist = tf.keras.datasets.fashion_mnist
( training_images, training_labels ), ( validation_images, validation_labels ) = mnist.load_data( )

# Now, THE FIRST CONVOLUTIONAL LAYER EXPECT EVERYTHING ALL TOGETHER, so we
# must reshape the whole list of 60000 images IN A SINGLE 4D TENSOR.
training_images		= training_images.reshape( 60000, 28, 28, 1 )
training_images 	= training_images / 255.0
# And do the same thing with the validation set.
validation_images	= validation_images.reshape( 10000, 28, 28, 1 )
validation_images 	= validation_images / 255.0

#####
#   #
# 2 # THE MODEL
#   #
#####
# Design the model
model = tf.keras.models.Sequential( [
  tf.keras.layers.Conv2D( 64, ( 3, 3 ), activation='relu', input_shape = ( 28, 28, 1 ) ),
  tf.keras.layers.MaxPooling2D( 2, 2 ),
  tf.keras.layers.Conv2D( 64, ( 3, 3 ), activation ='relu' ),
  tf.keras.layers.MaxPooling2D( 2, 2 ),
  tf.keras.layers.Flatten( ),
  tf.keras.layers.Dense( 20, activation = 'relu' ),
  tf.keras.layers.Dense( 10, activation = 'softmax' )
] )

# Build the model.
model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [ 'accuracy' ] )
model.summary( )

# Train the model.
model.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS )



################################################################################
# 
# I DO NOT FULLY UNDESTAND THIS PART DOWN HERE
#
################################################################################


def show_image(img):
  plt.figure()
  plt.imshow(validation_images[img].reshape(28,28))
  plt.grid(False)
  plt.show()  

f, axarr = plt.subplots(3,2)

print(validation_labels[:100])
# By scanning the list above I saw that the 0, 23 and 28 entries are all label 9 
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28

# For shoes (0, 23, 28), Convolution_Number=1 (i.e. the second filter) shows
# the sole being filtered out very clearly

CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]

activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

################################################################################
#
# HERE IT SEEMS TO BE GETTING THE OUTPUT OF SOME OF THE FILTERS.
#
################################################################################
for x in range(0,2):
  f1 = activation_model.predict(validation_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  
  f2 = activation_model.predict(validation_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  
  f3 = activation_model.predict(validation_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)


show_image(FIRST_IMAGE)
show_image(SECOND_IMAGE)
show_image(THIRD_IMAGE)

#### end of file ####
