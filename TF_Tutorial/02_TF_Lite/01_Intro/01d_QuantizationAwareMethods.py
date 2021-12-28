import tempfile
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import pathlib

# An evaluation function.
def evaluate_model(interpreter, test_images, test_labels):
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

	# Run predictions on every image in the "test" dataset.
	prediction_digits = []
	for i, test_image in enumerate(test_images):
		# Pre-processing: add batch dimension and convert to float32 to match with
		# the model's input data format.
		test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
		interpreter.set_tensor(input_index, test_image)

		# Run inference.
		interpreter.invoke()

		# Post-processing: remove batch dimension and find the digit with highest
		# probability.
		output = interpreter.tensor(output_index)
		digit = np.argmax(output()[0])
		prediction_digits.append(digit)

	print('\n')
	# Compare prediction results with ground truth labels to calculate accuracy.
	prediction_digits = np.array(prediction_digits)
	accuracy = (prediction_digits == test_labels).mean()

	return accuracy

export_dir = 'models/01d'
model_sizes 	= []
model_accuracy 	= []
model_labels 	= [	'ORIGINAL', 
					'QAWARE', 
					'TFLITE_ORIGINAL',
					'TFLITE_QAWARE']
#####
#   #
# 1 # THE DATA
#   #
#####
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

#####
#   #
# 2 # ORIGINAL MODEL
#   #
#####
# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the digit classification model
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_split=0.1,
)

folder = export_dir + '/original'
tf.saved_model.save(model, folder)

results = model.evaluate(test_images, test_labels)
model_accuracy.append(results[1])
model_sizes.append( pathlib.Path(folder + '/saved_model.pb').stat().st_size)

print("\n######################################################################")
print("ORIGINAL MODEL")
print("######################################################################")
model.summary()
print("######################################################################")
print("######################################################################\n")

#####
#   #
# 2 # QUATIZATION AWARE MODEL
#   #
#####
quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
q_aware_model.fit(
	train_images, 
	train_labels,
    #batch_size=500, 
    epochs=1, 
    validation_split=0.1)

folder = export_dir + '/q_aware'
tf.saved_model.save(q_aware_model, folder)

results = q_aware_model.evaluate(test_images, test_labels)
model_accuracy.append(results[1])
model_sizes.append( pathlib.Path(folder + '/saved_model.pb').stat().st_size)

print("\n######################################################################")
print("QUANTIZATION AWARE MODEL")
print("######################################################################")
q_aware_model.summary()
print("######################################################################")
print("######################################################################\n")

#####
#   #
# 3 # QUANTIZED MODEL
#   #
#####
modelLabels = ['ORIGINAL', 'QUANTIZATION_AWARE']
models = [model, q_aware_model]

for i in range(len(models)):
	print("MODEL OPTIMIZATION:", modelLabels[i])
	# Convert
	converter = tf.lite.TFLiteConverter.from_keras_model(models[i])
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	# Save model
	file = export_dir + modelLabels[i]
	tflite_model_file = pathlib.Path(file)	
	tflite_model_file.write_bytes(tflite_model)	

	# Load interpreter
	interpreter = tf.lite.Interpreter(model_content=tflite_model)
	interpreter.allocate_tensors()

	# Test
	results = evaluate_model(interpreter, test_images, test_labels)
	model_accuracy.append(results)
	model_sizes.append( pathlib.Path(file).stat().st_size)

print("\n######################################################################")
print("ACCURACY AND SIZES")
print("######################################################################")
for i in range(len(model_labels)):
	print("\n######################################################################")
	print(model_labels[i])
	print("Size:", model_sizes[i], "bytes")
	print("Accu:", round(model_accuracy[i],2), "%")
#### end of file ####
