import tensorflow as tf
import numpy as np

#####
#   #
# 1 # Import the dataset.
#   #
#####
# The dataset is one of the most important parts of ML. A good dataset properly
# labeled helps the ML model being well trained.
# Tensor flow has a good source of datasets to help us with our own TF learning
# process.
# Lets import handwritten dataset, mnist.
data = tf.keras.datasets.mnist
# Split the data in train (used to train our model ) and test ( to test 
# model performance )
( training_images, training_labels ), ( validation_images, validation_labels ) = data.load_data( )

#####
#   #
# 2 # Data preprocessing.
#   #
#####
# Data preprocessing is very important too. It modifies the data so we can
# input it to the model in the right way.
# Our dataset, mnist, comes in the form fo 0 - 255 dradient pixel images. Lets
# convert it into 0 - 1 floats.
training_images 	= training_images / 255.0
validation_images		= validation_images / 255.0

#####
#   #
# 3 # Design the NN
#   #
#####
model = tf.keras.models.Sequential( 
	[	tf.keras.layers.Flatten( input_shape = ( 28, 28 ) ),		  		# Input layer, 28x28, the size of the images, 
																																	# so a flat 784x1
		tf.keras.layers.Dense( 20, activation = tf.nn.relu ),				 	# Hiden layer with 20 neurons. The number needs to 
																																	# be not too bit, not too small. More on this later.
																																	# The relu activation function just set any value <0
																																	# back to 0. Useful for the model internal behaviour
		tf.keras.layers.Dense( 10, activation = tf.nn.softmax ) ] )		# Output layer with 10 neurons, one per digit.
																																	# The softmax function helps us to hightlight the
																																	# neuron with the highest value.

#####
#   #
# 4 # Tune the model up.
#   #
#####
model.compile( 	optimizer = 'adam',								# Adam varies the learning rate (gradients) in order to
																									# converge more quickly.
				loss = 'sparse_categorical_crossentropy',	# Loss function useful for classification.
				metrics = [ 'accuracy' ] )

#####
#   #
# 5 # Train the model.
#   #
#####
EPOCHS = 5
#model.fit( training_images, training_labels, epochs = EPOCHS )
#model.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS )
# We can stop the training after a certain "ccuracy" by using a callback like this:
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if( logs.get("accuracy") > 0.95):
      self.model.stop_training = True
callbacks = myCallback()

model.fit( training_images, training_labels, validation_data = ( validation_images, validation_labels ), epochs = EPOCHS, callbacks=[callbacks] )

#####
#   #
# 6 # Test the model.
#   #
#####
# Now that the model is trained, lest test it.
classifications = model.predict( validation_images )
predictions = []
for c in classifications:
	predictions.append( np.argmax( c ) )
print( "predictions: " + str( predictions[:40] ) )
print( "real thing : " + str( list(validation_labels[:40]) ) )

#####
#   #
# 7 # Evalueate the model.
#   #
#####
# This is already done during the fitting, but just for the sake of curiosity.
evaluation = model.evaluate(validation_images, validation_labels)
print( "\nModel Loss    : " + str( evaluation[ 0 ] ) )
print( "Model Accuracy: " + str( evaluation[ 1 ] ) )
