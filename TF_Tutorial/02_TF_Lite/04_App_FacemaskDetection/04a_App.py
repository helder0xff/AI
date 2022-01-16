# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-7-11-Assignment.ipynb
# A pre-trained model is a saved network that was previously trained on a large dataset, 
# typically on a large-scale image-classification task. You either use the pretrained model as is or
# use transfer learning to customize this model to a given task. 

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib

# CONSTANTS
BATCH_SIZE = 32
IMG_SIZE = (96, 96)
MODELS_DIR = 'models/'
EPOCHS = 5

#####
#	#
# 1 # THE DATA
#	#
#####
# PATHS
path_to_datasets = "../../../datasets/"
path_to_dataset  = os.path.join(os.path.dirname(path_to_datasets), 'edx_transfer_learningv3/edx_transfer_learning/')
train_dir = os.path.join(path_to_dataset, 'train')
validation_dir = os.path.join(path_to_dataset, 'validation')

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

# No test dataset, so take some of the images from the VALIDATION -> TEST,
# lets say, 20% of them (1/5)
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# ???????????????????????
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# AUGMENT DATA
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# RESCALE [0 - 255] -> [-1, 1]
preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# Note: Alternatively, you could rescale pixel values from[0,255]to[-1, 1]` using a Rescaling layer.
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1) 

#####
#	#
# 2 # THE MODEL
#	#
#####
# THE BASE MODEL
export_dir = MODELS_DIR + '/base'
if 	False == pathlib.Path(export_dir).is_dir():
	# Create the base model from the pre-trained model MobileNet V2
	IMG_SHAPE = IMG_SIZE + (3,)
	base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
	                                               include_top=False,
	                                               weights='imagenet')

	# FREEZE THE MODEL
	base_model.trainable = False

	# SAVE THE MODEL
	#tf.saved_model.save(base_model, export_dir)
	base_model.save(export_dir)
	base_model.summary()
else:
	#base_model = tf.saved_model.load(export_dir)
	base_model = tf.keras.models.load_model(export_dir)
	base_model.summary()

export_dir = MODELS_DIR + '/our'
if 	False == pathlib.Path(export_dir).is_dir():
	# ???????????????
	image_batch, label_batch = next(iter(train_dataset))
	feature_batch = base_model(image_batch) # Is this the output of the base model?

	# ADD FIRST LAYER
	# To begin the process of generating classifications from the pretrained features, 
	#we use a tf.keras.layers.GlobalAveragePooling2D layer to convert the 5x5 spatial features 
	# into a single 1024-element feature vector per image.
	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	feature_batch_average = global_average_layer(feature_batch)

	# ADD SECOND LAYER
	# We then apply a tf.keras.layers.Dense layer to convert the feature vector into a single prediction per image.
	prediction_layer = tf.keras.layers.Dense(1)
	prediction_batch = prediction_layer(feature_batch_average)

	# BUILD THE MODEL
	inputs = tf.keras.Input(shape=(96, 96, 3))	# INPUT 				LAYER
	x = data_augmentation(inputs)				# AUGMENTATION 			LAYER
	x = preprocess_input(x)						# [0-255] -> [-1, 1]	LAYER 
	x = base_model(x, training=False)			# BASE_MODEL 			LAYER
	x = global_average_layer(x)					# TRANSITION 			LAYER
	x = tf.keras.layers.Dropout(0.2)(x)			# WTF 					LAYER
	outputs = prediction_layer(x)				# PREDICTION 			LAYER
	model = tf.keras.Model(inputs, outputs)		# THE FINAL 			MODEL

	# COMPILE THE MODEL
	base_learning_rate = 0.0001
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
	              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=['accuracy'])

#####
#	#
# 3 # TRAIN THE MODEL
#	#
#####
	print("\n######################################################################")
	print("BEFORE OUR OWN TRAIN:")
	print("######################################################################")
	loss, accuracy = model.evaluate(validation_dataset)
	print("initial loss: {:.2f}".format(loss))
	print("initial accuracy: {:.2f}".format(accuracy))
	print("######################################################################")


	history = model.fit(train_dataset,
	                    epochs=EPOCHS,
	                    validation_data=validation_dataset)

	model.save(export_dir)
	model.summary()
	print("\n######################################################################")
	print("AFTER OUR OWN TRAIN:")
	print("######################################################################")
	loss, accuracy = model.evaluate(validation_dataset)
	print("initial loss: {:.2f}".format(loss))
	print("initial accuracy: {:.2f}".format(accuracy))
	print("######################################################################")
else:
	model = tf.keras.models.load_model(export_dir)
	model.summary()

print("\n######################################################################")
print("MODEL EVALUATION:")
print("######################################################################")
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
print("######################################################################")

#### end of file ####
