import tensorflow as tf
from tensorflow import keras
import numpy as np

#####
#   #
# 1 # Define and train the model.
#   #
#####
# One layer one neuron input
# -->[]-->
my_layer = keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([my_layer])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Get set ready, input + target
# This is the input and output of a function
# y(x) = w*x + b where w = 2 and b = -1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model.
model.fit(xs, ys, epochs=500)



#####
#   #
# 2 # Predict.
#   #
#####
print("\ny(10.0) prediction: " + str( model.predict( [ 10.0 ] )[ 0 ] ) )

#####
#   #
# 3 # Check the weights.
#   #
#####
param = my_layer.get_weights( )
print( "\nw = " + str(param[0]))
print( "b = " + str(param[1]))

#### end of file ####
