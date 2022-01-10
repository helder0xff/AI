################################################################################
# Based on:
# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-5-13-PretrainedModel.ipynb#scrollTo=g7eZJQUxn-Ri
################################################################################

import sys
import os
import pathlib
import numpy as np
import tensorflow.compat.v1 as tf
import librosa
import scipy.io.wavfile

# We add this path so we can import the SPEECH PROCESSING MODULES.
SPEECH_COMMANDS_PATH = '../../../tensorflow-2.4.1/tensorflow/examples/speech_commands/'
sys.path.append(SPEECH_COMMANDS_PATH)
import input_data
import models
import pickle

################################################################################
# CONSTANTS
################################################################################
KEYWORDS = "yes,no"

TRAINING_STEPS 	= "12000,3000"
LEARNING_RATE 	= "0.001,0.0001"

MODEL_ARCHITECTURE = 'tiny_conv'
# Ohter chooices could be single_fc, conv, low_latency_conv, low_latency_svdf, tiny_embedding_conv

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = KEYWORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20

# Constants used during training only
VERBOSITY = 'DEBUG'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR =  '../../../datasets/keywords/'
LOGS_DIR = 'logs_b/'
TRAIN_DIR = 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
MODELS_DIR = 'models_b'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# URL for the dataset and train/val/test split
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# Calculate the correct flattened input data shape for later use in model conversion
# since the model takes a flattened version of the spectrogram. The shape is number of 
# overlapping windows times the number of frequency bins. For the default settings we have
# 40 bins (as set above) times 49 windows (as calculated below) so the shape is (1,1960)
def window_counter(total_samples, window_size, stride):
  '''helper function to count the number of full-length overlapping windows'''
  window_count = 0
  sample_index = 0
  while True:
    window = range(sample_index,sample_index+stride)
    if window.stop < total_samples:
      window_count += 1
    else:
      break
    
    sample_index += stride
  return window_count

OVERLAPPING_WINDOWS = window_counter(CLIP_DURATION_MS, int(WINDOW_SIZE_MS), WINDOW_STRIDE)
FLATTENED_SPECTROGRAM_SHAPE = (1, OVERLAPPING_WINDOWS * FEATURE_BIN_COUNT)

# Print the configuration to confirm it
print("Training these words: %s" % KEYWORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)
################################################################################

def main():
	print("\n################################################################################")
	print("TRAINING THE MODEL")
	print("################################################################################")
	cmnd = "python {}/train.py \
		--data_dir={} \
		--wanted_words={} \
		--silence_percentage={} \
		--unknown_percentage={} \
		--preprocess={} \
		--window_stride={} \
		--model_architecture={} \
		--how_many_training_steps={} \
		--learning_rate={} \
		--train_dir={} \
		--summaries_dir={} \
		--verbosity={} \
		--eval_step_interval={} \
		--save_step_interval={}".format( SPEECH_COMMANDS_PATH,DATASET_DIR, KEYWORDS,	\
													SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE, \
													PREPROCESS, WINDOW_STRIDE, MODEL_ARCHITECTURE, \
													TRAINING_STEPS, \
													LEARNING_RATE, TRAIN_DIR, LOGS_DIR, \
													VERBOSITY, EVAL_STEP_INTERVAL, SAVE_STEP_INTERVAL)
	os.system(cmnd)
	print("################################################################################")
	
	print("\n################################################################################")
	print("LOADING PRE-TRAINED MODEL")
	print("################################################################################")
	if 	False == pathlib.Path(SAVED_MODEL).is_dir():
		cmnd = 'rm -rf ' + SAVED_MODEL
		os.system(cmnd)
		cmnd = "python {}/freeze.py\
				--wanted_words={} \
				--window_stride_ms={} \
				--preprocess={} \
				--model_architecture={} \
				--start_checkpoint={}{}'.ckpt-'{} \
				--save_format=saved_model \
				--output_file={}".format(SPEECH_COMMANDS_PATH, KEYWORDS, WINDOW_STRIDE, \
													PREPROCESS, MODEL_ARCHITECTURE, TRAIN_DIR, \
													MODEL_ARCHITECTURE, TOTAL_STEPS, SAVED_MODEL)
	print("################################################################################")

	print("\n################################################################################")
	print("GENERATING TENSOR FLOW LITE MODEL")
	print("################################################################################")
	# DEFINE AUDIO SETTINGS
	model_settings = models.prepare_model_settings(
	    len(input_data.prepare_words_list(KEYWORDS.split(','))),
	    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
	    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
	audio_processor = input_data.AudioProcessor(
	    DATA_URL, DATASET_DIR,
	    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
	    KEYWORDS.split(','), VALIDATION_PERCENTAGE,
	    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

	if 	False == pathlib.Path(FLOAT_MODEL_TFLITE).is_file() and \
		False == pathlib.Path(MODEL_TFLITE).is_file():
	# CREATE FLOAT AND QUANTIZED LITE MODELS
		with tf.Session() as sess:
		  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
		  float_tflite_model = float_converter.convert()
		  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
		  print("Float model is %d bytes" % float_tflite_model_size)

		  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
		  converter.optimizations = [tf.lite.Optimize.DEFAULT]
		  converter.inference_input_type = tf.lite.constants.INT8	
		  converter.inference_output_type = tf.lite.constants.INT8		 
		  def representative_dataset_gen():
		    for i in range(100):
		      data, _ = audio_processor.get_data(1, i*1, model_settings,
		                                         BACKGROUND_FREQUENCY, 
		                                         BACKGROUND_VOLUME_RANGE,
		                                         TIME_SHIFT_MS,
		                                         'testing',
		                                         sess)
		      flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
		      yield [flattened_data]
		  converter.representative_dataset = representative_dataset_gen
		  tflite_model = converter.convert()
		  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
		print("################################################################################")

	print("\n################################################################################")
	print("CHECKING ACCURACY")
	print("################################################################################")
	# Helper function to run inference
	def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
	  #
	  # Load test data
	  #
	  np.random.seed(0) # set random seed for reproducible test results.
	  with tf.Session() as sess:
	  # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
	    test_data, test_labels = audio_processor.get_data(
	        -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
	        TIME_SHIFT_MS, 'testing', sess)
	  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

	  #
	  # Initialize the interpreter
	  #
	  interpreter = tf.lite.Interpreter(tflite_model_path)
	  interpreter.allocate_tensors()
	  input_details = interpreter.get_input_details()[0]
	  output_details = interpreter.get_output_details()[0]
	  
	  #
	  # For quantized models, manually quantize the input data from float to integer
	  #
	  if model_type == "Quantized":
	    input_scale, input_zero_point = input_details["quantization"]
	    test_data = test_data / input_scale + input_zero_point
	    test_data = test_data.astype(input_details["dtype"])

	  #
	  # Evaluate the predictions
	  #
	  correct_predictions = 0
	  for i in range(len(test_data)):
	    interpreter.set_tensor(input_details["index"], test_data[i])
	    interpreter.invoke()
	    output = interpreter.get_tensor(output_details["index"])[0]
	    top_prediction = output.argmax()
	    correct_predictions += (top_prediction == test_labels[i])

	  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
	      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))	

	# Compute float model accuracy
	run_tflite_inference_testSet(FLOAT_MODEL_TFLITE)

	# Compute quantized model accuracy
	run_tflite_inference_testSet(MODEL_TFLITE, model_type='Quantized')
	
	print("################################################################################")

main()
	
#### end of file ####
