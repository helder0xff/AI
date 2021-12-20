import tensorflow as tf
from tensorflow import keras
import numpy as np

#####
#   #
# 1 # Define and train the model.
#   #
#####
# Two layer / three neurons
# -->[]-->
#			[]-->
# -->[]-->
my_layer_1 = keras.layers.Dense(units=2, input_shape=[1])
my_layer_2 = keras.layers.Dense(units=1)
model = tf.keras.Sequential([my_layer_1, my_layer_2])
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
# 3 # Play with the weights.
#   #
#####
# If we inspect the weights of the neurons, we can see that the second layer
# has two w. This is because they are the entrance from the first layer. The
# maths are done by the TF itself: do not panic.
param_1 = my_layer_1.get_weights( )
param_2 = my_layer_2.get_weights( )
layer1_w1=param_1[0][0][0]
layer1_w2=param_1[0][0][1]
layer1_b1=param_1[1][0]
layer1_b2=param_1[1][1]
layer2_w1 = param_2[0][0]
layer2_w2=param_2[0][1]
layer2_b=param_2[1][0]
print( "\n layer 1 params: " + str( param_1 ) )
print( "\nlayer1_w1 = " + str(layer1_w1))
print( "layer1_b1 = " + str(layer1_b1))
print( "\nlayer1_w2 = " + str(layer1_w2))
print( "layer1_b2 = " + str(layer1_b2 ))
print( "\n layer 2 params: " + str( param_2 ) )
print( "\nlayer2_w1 = " + str(layer2_w1 ))
print( "layer2_w2 = " + str(layer2_w2 ))
print( "layer2_b = " + str(layer2_b ))

# So if we want to do the magic ourselfs:
neuron1_out = (layer1_w1 * 10.0) + layer1_b1
neuron2_out = (layer1_w2 * 10.0) + layer1_b2 
neuron3_out = (layer2_w1 * neuron1_out) + (layer2_w2 * neuron2_out) + layer2_b
print( "\ny(x) = N3w1 * ( N1w * x + N1b ) + N3w2 *( N2w * x + N2b ) + N3b" )
print("y(10.0) manual prediction: " + str( neuron3_out ) )
print("y(10.0) model  prediction: " + str( model.predict( [ 10.0 ] )[ 0 ] ) )

#### end of file ####
