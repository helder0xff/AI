import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

#####
#	#
# 1 #	GET DATA
#	#
#####
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

#####
#	#
# 2 #	GET DATA READY
#	#
#####
# Pop the survived column out.
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

###
#a# Convert Data To Symbolic Parameters
###
# As there are different kind of data (strings, boolean, int, float, ...)
# we convert it all to symbolic parameters.

# Set dtype on the dat to string or float depending on if it is an object or it is not.
inputs = {}
for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32
  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

###
#b# Gather all numeric values together (tf.float) in a single input layer
###
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]

###
#c# Append string data to the preprocessed_inputs
### 
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = layers.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

###
#d# Concatanate all together
###
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

#####
#	#
# 2 #	BUILD A MODEL
#	#
#####
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
#tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

#### end of file ####
