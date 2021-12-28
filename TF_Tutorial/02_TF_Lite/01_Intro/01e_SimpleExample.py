import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tqdm import tqdm
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

#####
#   #
# 1 # THE DATA
#   #
#####
# Image conversion function
def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return  image, label

# Download the dataset
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'rock_paper_scissors', split=['train[:80%]', 'train[80%:]', 'test'], 
    with_info=True, as_supervised=True)
num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes

# Batch the data
BATCH_SIZE = 32
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

# Iterate through all the train batches, not really sure why
for image_batch, label_batch in train_batches.take(1):
    pass

#####
#   #
# 2 # THE BASELINE MODEL
#   #
#####
export_dir = './models/01e'
if False == pathlib.Path(export_dir).is_dir():
	# Take an already implemented model and fit to hour needs (size and feature vectors.)
	module_selection = ("mobilenet_v2", 224, 1280) 
	handle_base, pixels, FV_SIZE = module_selection
	MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
	IMAGE_SIZE = (pixels, pixels)
	print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

	# Get the layer's ready for hour model.
	feature_extractor = hub.KerasLayer(MODULE_HANDLE,
	                                   input_shape=IMAGE_SIZE + (3,), 
	                                   output_shape=[FV_SIZE],
	                                   trainable=False)

	# Fit the output layers to the number of classes we have.
	model = tf.keras.Sequential([
	        feature_extractor,
	        tf.keras.layers.Dense(num_classes, activation='softmax')
	])

	model.summary()

	model.compile(optimizer='adam',
	                  loss='sparse_categorical_crossentropy',
	                  metrics=['accuracy'])

	EPOCHS = 5

	hist = model.fit(train_batches,
	                 epochs=EPOCHS,
	                 validation_data=validation_batches)

	# Save Model
	tf.saved_model.save(model, export_dir)

#####
#   #
# 3 # CONVERT THE MODEL
#   #
#####
tflite_models_dir = pathlib.Path(export_dir)
tflite_model_file = tflite_models_dir/'tflite'
if False == pathlib.Path(tflite_model_file).is_file():
	converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
	tflite_model = converter.convert()

	# Save tf lite model
	tflite_model_file.write_bytes(tflite_model)
	# This will report back the file size in bytes
	# you will note that this model is too big for our Arduino
	# but would work on a mobile phone

#####
#   #
# 4 # LOAD MODEL INTO AN INTERPRETER
#   #
#####
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
predictions = []

# Split labels and images
test_labels, test_images = [], []
for img, label in tqdm(test_batches.take(100)):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))    
    test_labels.append(label.numpy()[0])
    test_images.append(img)
# This will report how many iterations per second, where each
# iteration is 100 predictions

# This will tell you how many of the predictions were correct
score = 0
for item in range(0,99):
  prediction=np.argmax(predictions[item])
  label = test_labels[item]
  if prediction==label:
    score=score+1

print("\nOut of 100 predictions I got " + str(score) + " correct")

#### end of file ####
