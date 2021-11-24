import tensorflow as tf
import numpy as np
from tensorflow import keras

# DEFINE A SINGLE LAYER SINGLE NEURON LAYER

#####
#   #
# 1 # Define NN
#   #
#####
# To the sequential method we introduce a LIST of layers, in our case just one.
# To the first layer we have to give the input_shape too. In our case [ 1 ], but in 
# the case of a 21x21 image, it would be ( 21, 21 )
model = keras.Sequential( [ keras.layers.Dense( units = 1, input_shape = [ 1 ] ) ] )

#####
#   #
# 2 # Define model
#   #
#####
# Then we give the model optimizer and the loss function.c
# In this particular case the optimizer is 'sgd': stocastich gradient descent.
model.compile( optimizer = 'sgd', loss = 'mean_squared_error' )

#####
#   #
# 3 # Fit the guy
#   #
#####
# The inputs must be in the form of a numpy array.
xs = np.array( [ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 ], dtype = float )
ys = np.array( [ -3.0, -1.0, 1.0, 3.0, 5.0, 7.0 ], dtype = float )
# The epochs, is the number of times the guy iterate to try to find
# the minimun of the loss function.
model.fit( xs, ys, epochs = 500 )

print( model.predict( [ 10.0, 4.0 ] ) )
