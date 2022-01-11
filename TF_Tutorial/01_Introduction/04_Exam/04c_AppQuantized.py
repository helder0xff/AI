"""
This script analize the accuracy of a given NN with the obtained datasets from
the 04a_* script.
"""

import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2
from keras.preprocessing import image
import pathlib
import tensorflow_model_optimization as tfmot
import copy

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
parser.add_argument( 	'-r',
						'--remove',
						help = 'Remove existent models?', 
						default = 'False', 
						type = str, 
						choices = ['False', 'True'], 
						required = False )
args = parser.parse_args( )

EPOCHS      = args.epochs
STREAMMODE 	= args.streammode

if 'image' == STREAMMODE:
	CHANNELS = 'original'
else:
	CHANNELS 	= args.channels

print("\n################################################################################")
print("SCRIPT CONFIG")
print("epochs:", EPOCHS)
print("streammode:", STREAMMODE)
print("channels:", CHANNELS)
print("################################################################################")

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

# MODEL PATHS
EXPORT_DIR = './models/'
model_sizes 	= []
model_accuracy 	= []
model_labels 	= [	'ORIGINAL', 
					'QAWARE', 
					'TFLITE_ORIGINAL',
					'TFLITE_QAWARE']

if 'True' == args.remove:
	cmnd = 'rm -rf ' + EXPORT_DIR
	os.system(cmnd)

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

	test_dataset 	= []
	test_labels	= []
	index = 0
	for flower in flowers:
		path = test_path + '/' + flower + '/'
		for file in os.listdir( path ):
			npy = np.load( path + file )
			test_dataset.append( npy / 255.0 )
			test_labels.append( index )
		index += 1
	test_dataset 	= np.array(test_dataset)			
	test_labels 	= np.array(test_labels)

#####
#	#
# 2 # THE ORIGINAL MODEL
#	#
#####
print("\n################################################################################")
print("ORIGINAL MODEL")
print("################################################################################")
model_nn = tf.keras.models.Sequential([
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


export_dir = EXPORT_DIR + 'ORIGINAL/'
if 	False == pathlib.Path(export_dir).is_dir():
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

	model.save(export_dir)

	# Some NN info.
	model.summary()
print("################################################################################")

#####
#	#
# 2 # THE Q_AWARE MODEL
#	#
#####
print("\n################################################################################")
print("QAWARE MODEL")
print("################################################################################")
export_dir = EXPORT_DIR + 'QAWARE/'
if 	False == pathlib.Path(export_dir).is_dir():
	quantize_model = tfmot.quantization.keras.quantize_model

	q_aware_model = quantize_model(model_nn)

	# `quantize_model` requires a recompile.
	q_aware_model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
	              metrics=['accuracy'])

	# Train the model
	if 'image' == STREAMMODE:
		history = q_aware_model.fit(
		      training_generator, 
		      epochs = EPOCHS,
		      verbose = 1,
		      validation_data = validation_generator)
	else:
		history = q_aware_model.fit(
			training_dataset,
			training_labels,
			epochs = EPOCHS,
			validation_data = (validation_dataset, validation_labels)
			)

	tf.saved_model.save(q_aware_model, export_dir)

	# Some NN info.
	q_aware_model.summary()

print("################################################################################")	

#####
#	#
# 2 # THE QUANTIZED MODEL
#	#
#####
export_file = EXPORT_DIR+ 'tflite'
if 	False == pathlib.Path(export_file).is_file():
	export_dir = EXPORT_DIR + 'QAWARE/'
	converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	tflite_model_file = pathlib.Path(export_file)	
	tflite_model_file.write_bytes(tflite_model)	

#####
#	#
# 3 # TEST
#	#
#####
# ORIGINAL
export_dir = EXPORT_DIR + 'ORIGINAL/'
model = tf.keras.models.load_model(export_dir)
predictions = []
score = 0
length = len(test_dataset)
for i in range(length):
	data = test_dataset[i]
	prediction = np.argmax(model.predict( tf.expand_dims(data, axis=0)))
	predictions.append(prediction)
	if test_labels[i] == prediction:
		score += 1
	print(i + 1, '/', length, end = '\r')

print('\n#######################################################################')
print("ORIGINAL TEST INFO.")
print(score * 100 / length, "% accuracy")
print('#######################################################################')

# QUANTIZED
# Load quantized model
export_file = EXPORT_DIR+ 'tflite'
interpreter = tf.lite.Interpreter(model_path=export_file)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
score = 0
length = len(test_dataset)
for i in range(length):
	data = test_dataset[i]
	data = np.expand_dims(data, axis=0).astype(np.float32)
	interpreter.set_tensor(input_index, data)
	interpreter.invoke()
	prediction = np.argmax(interpreter.get_tensor(output_index))
	predictions.append(prediction)
	if test_labels[i] == prediction:
		score += 1
	print(i + 1, '/', length, end = '\r')

print('\n#######################################################################')
print("QUANTIZED TEST INFO.")
print(score * 100 / length, "% accuracy")
print('#######################################################################')

#### end of file ####
