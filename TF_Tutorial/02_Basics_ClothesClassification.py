# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#####
#	#
# 1 #	IMPORT DATASET
#	#
#####
# Import data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define classes (for later use)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Some data exploration (might be usefull to have this knowledge at some point )
# CMD: train_images.shape
# RET: ( length, width, height )

#####
#	#
# 2 #	IMAGE PREPROCESSING
#	#
#####
# The images are in the form of an integer gray scale [0 - 255]. We need to adjust
# this to be a float number [0 - 1]
train_images = train_images / 255.0
test_images = test_images / 255.0


#####
#	#
# 3 #	BUILD THE MODEL
#	#
#####
# The first layer converts the 28x28 matrix into a single array of 784 cells.
# This layer just reformat the data.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),	# Input NN, 28x28 pixel images.
    tf.keras.layers.Dense(128, activation='relu'),	
    tf.keras.layers.Dense(10)						# Output, 10 clothe classes.
])


#####
#	#
# 4 #	COMPILE THE MODEL
#	#
#####
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',		# how the model is updated
              loss=loss_fn,			# measures how accurate the model is during training
              metrics=['accuracy'])	# Used to monitor the training and testing steps


#####
#	#
# 5 #	TRAIN  THE MODEL
#	#
#####
model.fit( 	train_images, 	# input images
			train_labels, 	# labels
			epochs=10)		# iterations, from init epoch = 0 to epochs = 10


#####
#	#
# 6 #	EVALUATE THE MODEL
#	#
#####
# If test accuracy is lower to the train accuracy is due to overfitting!!!!
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print( "Test accuracy: " + str( test_acc ) )

#####
#	#
# 7 #	ADD LAYER TO THE MODEL
#	#
#####
# This layer converts the logits to probabilities.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


#####
#	#
# 8 #	PREDICT
#	#
#####
# Pick an image from the test dataset
pic_N = 0
testImage = test_images[ pic_N ]

# IMPORTANT: we have to convert it to a single picture dataset,
# as our model work on data sets, no single images.
testImage = (np.expand_dims(testImage,0))

# Make predictions for the chosen test image.
prediction = probability_model.predict(testImage)
print("\n######################################################################")
print("Prediction for one of the images:")
print(prediction)
print("######################################################################")
print("It seems to be a " + class_names[np.argmax( prediction )])
print("And it is a .... " + class_names[test_labels[pic_N]])
print("######################################################################\n")

#### end of file ####
