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
export_dir = 'models/01b'
tf.saved_model.save(model, export_dir)

#####
#   #
# 3 # SQUEEZE THE MODEL
#   #
#####
# Squeez
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save
tflite_model_file = pathlib.Path(export_dir + '/tflite')
tflite_model_file.write_bytes(tflite_model)

#####
#   #
# 4 # USE THE LITE MODEL
#   #
#####
# Load interpreter
#interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Some lite model info
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
print('#######################################################################')
print("LITE MODEL INFO.")
print('input:', input_index)
print('output:', output_index)
print('#######################################################################')

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

print('#######################################################################')
print("TEST INFO.")
print("Out of 100 predictions I got " + str(score) + " correct")
print('#######################################################################')

#### end of file ####
