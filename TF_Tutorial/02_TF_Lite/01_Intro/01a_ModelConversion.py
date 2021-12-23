import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pathlib

#####
#	#
# 1 # THE DATA
#	#
#####
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#####
#	#
# 2 # MODEL DESIGN
#	#
#####

# Design
layer = Dense(units=1, input_shape=[1])
model = Sequential([layer])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Fit
model.fit(xs, ys, epochs=500)

# Save the model
export_dir = 'models/01a'
tf.saved_model.save(model, export_dir)

print("########## BIG MODEL PREDICTION ##########")
x = 10.0
print("Model prediction for", x, model.predict([x]))
print("Here is what I learned(my weights): {}".format(layer.get_weights()))
print("##########################################")

#####
#	#
# 3 # MODEL CONVERSION
#	#
#####
# Convert the saved model to TF Lite
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the TF model
tflite_model_file = pathlib.Path(export_dir + '/tflite')
tflite_model_file.write_bytes(tflite_model)

#####
#	#
# 4 # LITE MODEL EVALUATION
#	#
#####
# We do not have a HW (yet), but TF Lite provide an itnerpreter. 
# Lets load the lite model into the interpreter and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Lets get some info. about the Lite model (the tensors)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("########## TINY MODEL DETAILS ##########")
print(input_details)
print(output_details)
print("########################################")

# Evaluate it
# Convert the input, x, to a tensor.
xTensor = np.array([[x]], dtype=np.float32)
# Set input tensor
interpreter.set_tensor(input_details[0]['index'], xTensor)
# Run the tiny model
interpreter.invoke()
# Get output tensor
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print("########## TINY MODEL PREDICTION ##########")
print("Model prediction for", x, tflite_results)
print("###########################################")

#### end of file ####
