import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import pathlib
from tqdm import tqdm

#####
#   #
# 1 # THE DATA
#   #
#####
def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return  image, label

# DOWNLAD the DATASET and split in train, validation and test
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Printing out some info. from metadata
num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print('#######################################################################')
print('DATASET INFO.')
print('Examples:', num_examples)
print('Classes:', num_classes)
print('#######################################################################')

# Get BATCHES ready.
BATCH_SIZE = 32
# Shuffle the train set - formtat image - batch
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
# format image - batch
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
# format image
test_batches = raw_test.map(format_image).batch(1)

for image_batch, label_batch in train_batches.take(1):
    pass
print('#######################################################################')
print('BATCH INFO.')
print('Batch shape: ', image_batch.shape)
print('#######################################################################')

export_dir = 'models/01c'
if False == pathlib.Path(export_dir).is_dir():
	#####
	#   #
	# 2 # MODEL DESIGN
	#   #
	#####

	# DOWNLOAD MODULE
	handle_base         = "mobilenet_v2"  # Model
	pixels              = 224
	feature_vector_size = 1280            # Layer output
	MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
	IMAGE_SIZE = (pixels, pixels)
	FV_SIZE = feature_vector_size

	print('#######################################################################')
	print('MDOULE INFO.')
	print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))
	print('#######################################################################')

	# BUILD LAYER from moudle
	feature_extractor = hub.KerasLayer(MODULE_HANDLE,
	                                   input_shape=IMAGE_SIZE + (3,), 
	                                   output_shape=[FV_SIZE],
	                                   trainable=False)

	# DESIGN MODEL
	model = tf.keras.Sequential([
	        feature_extractor,
	        tf.keras.layers.Dense(num_classes, activation='softmax')
	])

	print('#######################################################################')
	print('MODEL INFO.')
	model.summary()
	print('#######################################################################')

	model.compile(  optimizer='adam',
	                loss='sparse_categorical_crossentropy',
	                metrics=['accuracy'])    

	# FIT MODEL
	EPOCHS = 1
	hist = model.fit(train_batches,
	                 epochs=EPOCHS,
	                 validation_data=validation_batches)

	# Save the model
	tf.saved_model.save(model, export_dir)

#####
#   #
# 3 # SQUEEZE THE MODEL
#   #
#####
liteModels = []
liteModelsFiles = []

# Apply optimization
optimizationsLabels = ['', 'DEFAULT']
optimizations = ['', tf.lite.Optimize.DEFAULT]
for i in range(len(optimizations)):
	converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
	converter.optimizations = [optimizations[i]]
	tflite_model = converter.convert()
	tflite_model_file = pathlib.Path(export_dir + '/tflite' + optimizationsLabels[i])	
	tflite_model_file.write_bytes(tflite_model)
	liteModels.append(tflite_model)
	liteModelsFiles.append(tflite_model_file)

# Apply Quantization
quantizationLabels 	= ['TFLITE_BUILTINS_INT8']
quantization 		= [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
def representative_data_gen():
	for input_value, _ in test_batches.take(100):
		yield [input_value]

for i in range(len(quantization)):
	converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
	converter.representative_dataset = representative_data_gen
	converter.target_spec.supported_ops = [quantization[i]]
	tflite_model = converter.convert()
	tflite_model_file = pathlib.Path(export_dir + '/tflite' + quantizationLabels[i])	
	tflite_model_file.write_bytes(tflite_model)
	liteModels.append(tflite_model)
	liteModelsFiles.append(tflite_model_file)

# Apply Optimization + Quantization
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset 	= representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
tflite_model_file = pathlib.Path(export_dir + '/tflite' + 'OPTI_QUANTI')	
tflite_model_file.write_bytes(tflite_model)
liteModels.append(tflite_model)
liteModelsFiles.append(tflite_model_file)

#####
#   #
# 4 # USE THE LITE MODEL
#   #
#####
scores = []
for i in range(len(liteModels)):
	interpreter = tf.lite.Interpreter(model_content=liteModels[i])
	interpreter.allocate_tensors()

	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

	# Use it
	predictions = []
	# This will report how many iterations per second, where each
	# iteration is 100 predictions
	test_labels, test_imgs = [], []
	for img, label in tqdm(test_batches.take(100)):
	    interpreter.set_tensor(input_index, img)
	    interpreter.invoke()
	    predictions.append(interpreter.get_tensor(output_index))
	    
	    test_labels.append(label.numpy()[0])
	    test_imgs.append(img)

	# Test it
	score = 0
	for item in range(0,len(predictions)):
	  prediction=np.argmax(predictions[item])
	  label = test_labels[item]
	  if prediction==label:
	    score=score+1
	scores.append(score)	

print('#######################################################################')
print("LITE MODELS SIZES")
for file in liteModelsFiles:
	print(file, "size", pathlib.Path(file).stat().st_size, "bytes.")
print('#######################################################################')

print('#######################################################################')
print("TEST INFO.")
for i in range(len(liteModelsFiles)):
	print(liteModelsFiles[i], scores[i] * 100 / 100, "accuracy.")
print('#######################################################################')

#### end of file ####
