########################################################################################################################
#
#  https://colab.research.google.com/github/tinyMLx/colabs/blob/master/4-4-8-Flatbuffers.ipynb#scrollTo=v9eegi_vtxW4
#
# In this file we are gonna interact with the model to see how can we manipulate
# it for the sake of curiosity.
#
# We will need:
#   * The 'flatc' compiler: that converts the model format stored in a text schema to Python accessor classes.  
#     (Follow rules for its instalation on the link above).
#   * The text schema: a text file that describes the layout of the model file format.
#   * The Flatbuffer library: Python library that the accessor classes rely on.
#
# So with the 'flatc' compiler we generate a Python library following an schema file. This generated library is held
# in the tflite folder.
#
########################################################################################################################

################################################################################
#
#   IMPORTS
#
################################################################################
import flatbuffers
import sys
import numpy as np
sys.path.append('tflite/')
import Model
sys.path.append("tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models
import tensorflow as tf
################################################################################
################################################################################

################################################################################
#
#   HELPER FUNCTIONS
#
################################################################################
def load_model_from_file(model_filename):
    """
    This function load a *.tflite file model into a Model object.
    """
    with open(model_filename, "rb") as file:
        buffer_data = file.read()
    model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
    model = Model.ModelT.InitFromObj(model_obj)

    return model

def save_model_to_file(model, model_filename):
    """
    This function save a Model object into a *.tflite file.
    """
    # Get a buffer for the model.
    builder = flatbuffers.Builder(1024)
    # Pack the model into the buffer.
    model_offset = model.Pack(builder)
    # Close the buffer with the file identifier.
    builder.Finish(model_offset, file_identifier=b'TFL3')
    # Get the data to write to the file.
    model_data = builder.Output()
    # Write the data to the file.
    with open(model_filename, 'wb') as out_file:
        out_file.write(model_data)


def test_model_accuracy(model_filename):
    """
    This function test the model accuracy.
    """
    # Get test data and labels.
    with tf.compat.v1.Session() as sess:
        test_data, test_labels = audio_processor.get_data(-1, 0, model_settings,
                                                          0, 0, 0, 'testing', 
                                                          sess)
    # Load interpreter from the model.
    interpreter = tf.lite.Interpreter(model_filename)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    model_output = interpreter.tensor(output_index)

    correct_predictions = 0
    for i in range(len(test_data)):
        current_input = test_data[i]
        current_label = test_labels[i]
        flattened_input = np.array(current_input.flatten(), dtype=np.float32).reshape(1, 1960)
        interpreter.set_tensor(input_index, flattened_input)
        interpreter.invoke()
        top_prediction = model_output()[0].argmax()
        if top_prediction == current_label:
            correct_predictions += 1

    return (correct_predictions * 100) / len(test_data)
    
################################################################################
################################################################################

################################################################################
#
#   GET SYSTEM READY
#
################################################################################
# A comma-delimited list of the words you want to train for.
# The options are: yes,no,up,down,left,right,on,off,stop,go
# All the other words will be used to train an "unknown" label and silent
# audio data with no spoken words will be used to train a "silence" label.
WANTED_WORDS = "yes,no"

# The number of steps and learning rates can be specified as comma-separated
# lists to define the rate at each stage. For example,
# TRAINING_STEPS=12000,3000 and LEARNING_RATE=0.001,0.0001
# will run 12,000 training loops in total, with a rate of 0.001 for the first
# 8,000, and 0.0001 for the final 3,000.
TRAINING_STEPS = "12000,3000"
LEARNING_RATE = "0.001,0.0001"

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE =20
MODEL_ARCHITECTURE = 'tiny_conv' # Other options include: single_fc, conv,
                      # low_latency_conv, low_latency_svdf, tiny_embedding_conv

# Constants used during training only
VERBOSITY = 'WARN'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR = 'dataset/'
LOGS_DIR    = 'logs/'
TRAIN_DIR   = 'train/' # for training checkpoints and other files.

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

################################################################################
################################################################################

def main():
    # Load the model
    model = load_model_from_file('speech_commands_model/speech_commands_model_float.tflite')

    # Loop through all the weights held in the model.
    for buffer in model.buffers:
        # Skip missing or small weight arrays.
        if buffer.data is not None and len(buffer.data) > 1024:
            # Pull the weight array from the model, and cast it to 32-bit floats since
            # we know that's the type for all the weights in this model. In a real
            # application we'd need to check the data type from the tensor information
            # stored in the model.subgraphs
            original_weights = np.frombuffer(buffer.data, dtype=np.float32)

            # Modify the weights
            # Play with it
            # munged_weights = np.add(original_weights, 0.002)
            munged_weights = np.round(original_weights * (1/0.05)) * 0.05

            # Write the altered data back into the model.    
            buffer.data = munged_weights.tobytes()

    # Save the Model back.
    save_model_to_file(model, 'speech_commands_model/speech_commands_model_modified.tflite')        

    # Test models Accuracy
    print('Original accuracy:', test_model_accuracy('speech_commands_model/speech_commands_model_float.tflite'))
    print('Modified accuracy:', test_model_accuracy('speech_commands_model/speech_commands_model_modified.tflite'))

main()
#### end of file ####
