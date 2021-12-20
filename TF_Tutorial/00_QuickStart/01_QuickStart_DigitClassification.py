import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Helper libraries
import numpy as np

#####
#	#
# 1 #	IMPORT DATASET
#	#
#####
# Load MNSIT dataset. Dataset of handwritten digits: 0, 1, 2, ..., 9.
mnist = tf.keras.datasets.mnist

# Convert sample data from int -> float
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#####
#	#
# 2 #	BUILD THE MODEL
#	#
#####
# Build model. In this case a model of 4 sequencial NN layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),	# 28x28 pixel image
  tf.keras.layers.Dense(128, activation='relu'),	# ?
  tf.keras.layers.Dropout(0.2),						# ?
  tf.keras.layers.Dense(10)							# Output layer, one cell per digit.
])

#####
#	#
# 3 #	COMPILE(INITIALIZE) THE MODEL
#	#
#####
# Define loss function
# In this loss function loss is zero if the model is sure of the correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#####
#	#
# 4 #	TRAIN THE MODEL
#	#
#####
# Train
model.fit(	x_train,	# Input data, 28x28 pixel image
			y_train, 	# Target data, digit (0, 1, 2, ..., 9)
			epochs=5)	# Final iteration given the initial on, set to 0 by default.

#####
#	#
# 5 #	EVALUATE THE MODEL
#	#
#####
print("\n######################################################################")
print("How good the model is:")
model.evaluate(x_test,  y_test, verbose=2)
print("######################################################################\n")

#####
#	#
# 6 #	PREDICT
#	#
#####
pic_N = 7 # Image number to predict
# Pass an image to the model.
# This retunns an array of predictions with unormalize scores, s(x)),
# one for each class (0, 1, 2, ..., 9).
# [ s(0), 		s(1), 		  s(2),		  s(3),			s(4), 
#	s(5), 		s(6), 		  s(7), 	  s(8),			s(9) ]
# [ 0.74148655, -0.39261633,  0.08016336, -0.46431944,  0.21458861,
# 	0.31183302,  0.7555975 ,  0.80728006, -0.6296631 , -0.4926056 ]
print("\n######################################################################")
print("Prediction:")
predictions = model(x_train[pic_N:pic_N + 1]).numpy()
print(predictions)
print("######################################################################\n")

# Normalize the predictions -> get probabilities, p(x)
# [	p(0), 		p(1), 		p(2), 		p(3), 		p(4), 
#	p(5), 		p(6), 		p(7), 		p(8), 		p(9) ]
# [	0.16651046, 0.05356818, 0.08594736, 0.04986165, 0.09831339,
#  	0.10835411, 0.16887674, 0.1778342 , 0.04226285, 0.04847102]
print("\n######################################################################")
print("Normalize prediction:")
predictions = tf.nn.softmax(predictions).numpy()
print(predictions)
print("######################################################################\n")

print("\n######################################################################")
print("Our prediction is: " + str( np.argmax( predictions ) ) )
print("The pic. label is: " + str( y_train[pic_N] ) )
print("######################################################################\n")

#### end of file ####
