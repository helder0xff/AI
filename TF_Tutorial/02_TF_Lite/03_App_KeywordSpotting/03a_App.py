################################################################################
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

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels 						= KEYWORDS.count(',') + 1
number_of_total_labels 					= number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples 	= int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE 						= equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE 						= equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS 			= 'micro'
WINDOW_STRIDE 		= 20			# Window width
MODEL_ARCHITECTURE 	= 'tiny_conv'

# Constants for training directories and filepaths
DATASET_DIR 	= '../../../datasets/keywords/'
LOGS_DIR 		= 'logs/'
TRAIN_DIR 		= 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF 			= os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE 		= os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE 	= os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO 	= os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL 		= os.path.join(MODELS_DIR, 'saved_model')

# Constants for Quantization
QUANT_INPUT_MIN 	= 0.0
QUANT_INPUT_MAX 	= 26.0
QUANT_INPUT_RANGE 	= QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE 			= 16000
CLIP_DURATION_MS 		= 1000
WINDOW_SIZE_MS 			= 30.0
FEATURE_BIN_COUNT 		= 40
BACKGROUND_FREQUENCY 	= 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS 			= 100.0

# URL for the dataset and train/val/test split
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# Pre-trained model constants
TOTAL_STEPS = 15000 # used to identify which checkpoint file
################################################################################
################################################################################


def main():
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
		os.system(cmnd)
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

	print("\n################################################################################")
	print("TESTING")
	print("################################################################################")
	fid = open('yes_no.pkl', 'rb')
	audio_files = pickle.load(fid)
	yes1 = audio_files['yes1']
	yes2 = audio_files['yes2']
	yes3 = audio_files['yes3']
	yes4 = audio_files['yes4']
	no1 = audio_files['no1']
	no2 = audio_files['no2']
	no3 = audio_files['no3']
	no4 = audio_files['no4']
	sr_yes1 = audio_files['sr_yes1']
	sr_yes2 = audio_files['sr_yes2']
	sr_yes3 = audio_files['sr_yes3']
	sr_yes4 = audio_files['sr_yes4']
	sr_no1 = audio_files['sr_no1']
	sr_no2 = audio_files['sr_no2']
	sr_no3 = audio_files['sr_no3']
	sr_no4 = audio_files['sr_no4']	

	# Helper function to run inference (on a single input this time)
	# Note: this also includes additional manual pre-processing
	TF_SESS = tf.InteractiveSession()
	def run_tflite_inference_singleFile(tflite_model_path, custom_audio, sr_custom_audio, model_type="Float"):
		#
		# Preprocess the sample to get the features we pass to the model
		#
		# First re-sample to the needed rate (and convert to mono if needed)
		custom_audio_resampled = librosa.resample(librosa.to_mono(np.float64(custom_audio)), sr_custom_audio, SAMPLE_RATE)
		# Then extract the loudest one second
		scipy.io.wavfile.write('custom_audio.wav', SAMPLE_RATE, np.int16(custom_audio_resampled))
		cmnd = '/tmp/extract_loudest_section/gen/bin/extract_loudest_section custom_audio.wav ./trimmed'
		os.system(cmnd)
		#!/tmp/extract_loudest_section/gen/bin/extract_loudest_section custom_audio.wav ./trimmed
		# Finally pass it through the TFLiteMicro preprocessor to produce the 
		# spectrogram/MFCC input that the model expects
		custom_model_settings = models.prepare_model_settings(
		  	0, SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
		  	WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
		custom_audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0,
		                                                model_settings, None)
		custom_audio_preprocessed = custom_audio_processor.get_features_for_wav(
		                                    'trimmed/custom_audio.wav', model_settings, TF_SESS)
		# Reshape the output into a 1,1960 matrix as that is what the model expects
		custom_audio_input = custom_audio_preprocessed[0].flatten()
		test_data = np.reshape(custom_audio_input,(1,len(custom_audio_input)))

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
		# Run the interpreter
		#
		interpreter.set_tensor(input_details["index"], test_data)
		interpreter.invoke()
		output = interpreter.get_tensor(output_details["index"])[0]
		top_prediction = output.argmax()

		#
		# Translate the output
		#
		top_prediction_str = ''
		if top_prediction == 2 or top_prediction == 3:
			top_prediction_str = KEYWORDS.split(',')[top_prediction-2]
		elif top_prediction == 0:
			top_prediction_str = 'silence'
		else:
			top_prediction_str = 'unknown'

		print('%s model guessed the value to be %s' % (model_type, top_prediction_str))
		return top_prediction_str

	print("Testing yes1")
	run_tflite_inference_singleFile(MODEL_TFLITE, yes1, sr_yes1, model_type="Quantized")
	print("Testing yes2")
	run_tflite_inference_singleFile(MODEL_TFLITE, yes2, sr_yes2, model_type="Quantized")
	print("Testing yes3")
	run_tflite_inference_singleFile(MODEL_TFLITE, yes3, sr_yes3, model_type="Quantized")
	print("Testing yes4")
	run_tflite_inference_singleFile(MODEL_TFLITE, yes4, sr_yes4, model_type="Quantized")
	print("Testing no1")
	run_tflite_inference_singleFile(MODEL_TFLITE, no1, sr_no1, model_type="Quantized")
	print("Testing no2")
	run_tflite_inference_singleFile(MODEL_TFLITE, no2, sr_no2, model_type="Quantized")
	print("Testing no3")
	run_tflite_inference_singleFile(MODEL_TFLITE, no3, sr_no3, model_type="Quantized")
	print("Testing no4")
	run_tflite_inference_singleFile(MODEL_TFLITE, no4, sr_no4, model_type="Quantized")
	
	print("################################################################################")

main()
#### end of file ####	