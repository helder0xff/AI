# https://colab.research.google.com/github/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/arduino_tinyml_workshop.ipynb#scrollTo=AGChd1FAk5_j

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pathlib

# CONSTANTS
SEED = 1234
GESTURES = ["punch", "flex"]
NUM_GESTURES = len(GESTURES)
SAMPLES_PER_GESTURE = 119
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)
DELETE = True

#####
#   #
# 1 # GET THE DATA
#   #
#####
labels 	= []
inputs 	= []
for gesture_index in range(NUM_GESTURES):
	# Get the gesture
	gesture = GESTURES[gesture_index]
	print('Processing gesture', gesture)

	# Read the gesture data
	label = ONE_HOT_ENCODED_GESTURES[gesture_index]
	df = pd.read_csv(gesture + ".csv")

	num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

	for i in range(num_recordings):
		tensor = []
		for j in range(SAMPLES_PER_GESTURE):
			index = i * SAMPLES_PER_GESTURE + j
			# normalize the input data, between 0 to 1:
			# - acceleration is between: -4 to +4
			# - gyroscope is between: -2000 to +2000
			tensor += 	[	(df['aX'][index] + 4) / 8,
          					(df['aY'][index] + 4) / 8,
          					(df['aZ'][index] + 4) / 8,
          					(df['gX'][index] + 2000) / 4000,
          					(df['gY'][index] + 2000) / 4000,
          					(df['gZ'][index] + 2000) / 4000]

		inputs.append(tensor)
		labels.append(label)
inputs = np.array(inputs)
labels = np.array(labels)

#####
#   #
# 2 # SPLIT THE DATA
#   #
######
# Randomize input order
num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

inputs = inputs[randomize]
labels = labels[randomize]

TRAIN_INDEX = int(0.6 * num_inputs)
TEST_INDEX = int(0.2 * num_inputs + TRAIN_INDEX)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_INDEX, TEST_INDEX])
labels_train, labels_test, labels_validate = np.split(labels, [TRAIN_INDEX, TEST_INDEX])

#####
#   #
# 3 # THE MODEL
#   #
#####
export_dir = 'model'
if 	False == pathlib.Path(export_dir).is_dir() or True == DELETE:
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=(714, 1)))
	model.add(tf.keras.layers.Dense(50, activation='relu')) # relu is used for performance
	model.add(tf.keras.layers.Dense(15, activation='relu'))
	model.add(tf.keras.layers.Dense(NUM_GESTURES, activation='softmax')) # softmax is used, because we only expect one gesture to occur per input

	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	history = model.fit(inputs_train, labels_train, epochs=100, batch_size=1, validation_data=(inputs_validate, labels_validate))
	model.save(export_dir)

#####
#   #
# 4 # PREDICT
#   #
#####
# use the model to predict the test inputs
model = tf.keras.models.load_model(export_dir)
print(model.summary())
predictions = model.predict(inputs_test)

for i in range(len(predictions)):
	print("#########")
	if predictions[i].argmax() == labels_test[i].argmax():
		print("RIGHT")
	else:
		print("WRONG")

#####
#   #
# 5 # CONVERT THE MODEL
#   #
#####
tflite_file = "gesture_model.tflite"
if 	False == pathlib.Path(tflite_file).is_file() or True == DELETE:
	# Convert the model to the TensorFlow Lite format without quantization
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()

	# Save the model to disk
	open("gesture_model.tflite", "wb").write(tflite_model)
	cmnd = "echo 'const unsigned char model[] = {' > IMU_Classifier/model.h"
	os.system(cmnd)
	cmnd = "cat gesture_model.tflite | xxd -i >> IMU_Classifier/model.h"
	os.system(cmnd)
	cmnd = "echo '};' >> IMU_Classifier/model.h"
	os.system(cmnd)		  
	basic_model_size = os.path.getsize(tflite_file)
	print("Model is %d bytes" % basic_model_size)

#### end of file ####
