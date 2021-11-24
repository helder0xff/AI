import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#####
#	#
# 1 #	GET DATA
#	#
#####
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

#####
#	#
# 2 #	GET DATA READY
#	#
#####
# As we want to predict the age, pop it out.
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# Convert to array
abalone_features 	= np.array( abalone_features )
abalone_labels 		= np.array( abalone_labels )

# Save one of the abalones for testing 
abalone_f = abalone_features[:1]
abalone_l = abalone_labels[:1]
abalone_features = abalone_features[1:]
abalone_labels = abalone_labels[1:]

#####
#	#
# 3 #	BUILD THE MODEL
#	#
#####
# Get the basic model
abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

# Get a normalization layer at the doorway
normalize = layers.Normalization()
normalize.adapt( abalone_features )

# Plug it into the basic model
norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

# Compile the model
loss_fn = tf.losses.MeanSquaredError()
abalone_model.compile( loss = loss_fn, optimizer = tf.optimizers.Adam() )

#####
#	#
# 4 #	FIT (TRAIN) THE MODEL
#	#
#####
abalone_model.fit(abalone_features, abalone_labels, epochs=10)

prediction = abalone_model.predict( np.array( abalone_f ) )
print( "Prediction:" + str( prediction[0] ) )
print( "Real age  :" + str( abalone_l ) )

#### end of file ####
