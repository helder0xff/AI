# In this script we will see how to use the loss function to optimize
# our guess.
# Basically we have a linear function y = a * x + b and we want to find
# what are the a and b parameters that make the equation match a given output.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#####
#   #
# 1 # Define our simple linear regression model.
#   #
#####
# We have a and b as class members, and
# __call__, the function that makes our equation up.
# When calling the model as follow: model( inputs ), where inputs is a list,
# it returns a list of the outputs.
class Model(object):
  def __init__( self, initial_A, initial_B, learning_rate ):
    # Initialize the weights
    self.a = tf.Variable(initial_A)
    self.b = tf.Variable(initial_B)
    self.learning_rate = learning_rate

  def getLearningRate( self ):
    return self.learning_rate

  def __call__(self, x):
    return self.a * x + self.b

#####
#   #
# 2 # Define our loss function. 
#   #
#####
def loss( predicted_y, target_y ):
  return tf.reduce_mean( tf.square( predicted_y - target_y ) )

#####
#   #
# 3 # Define our training procedure.
#   #
#####
# a) Get current loss
# b) Differentiate a and b regarding the loss.
# c) Update a and b
def train( model, inputs, outputs ):
  with tf.GradientTape() as t:
    # current_loss = loss( model( inputs ) = outputs of running __call__, the target outputs )
    current_loss = loss( model( inputs ), outputs)

    # Here is where you differentiate the model values with respect to the loss function
    da, db = t.gradient( current_loss, [model.a, model.b] )
    
    # And here is where you update the model values based on the learning rate chosen
    learning_rate = model.getLearningRate( )
    model.a.assign_sub( learning_rate * da )
    model.b.assign_sub( learning_rate * db )
    
    return current_loss

def main( ):
  # Entries and outputs for the seek a and b.
  xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
  ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
  
  # Instantiate model
  model = Model( initial_A = 10.0, initial_B = 10.0, learning_rate = 0.09 )
  
  # Train
  list_a, list_b = [ ], [ ]
  losses = [ ]
  epochs = range( 50 )
  for epoch in epochs:
    # Just append data to plot the learning process.
    list_a.append( model.a.numpy( ) )
    list_b.append( model.b.numpy( ) )
    # Tran the model with the data.
    losses.append( train( model, xs, ys ) )

  TRUE_a = 2.0
  TRUE_b = -1.0
  plt.plot(epochs, list_a, 'r', epochs, list_b, 'b')
  plt.plot([TRUE_a] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
  plt.legend(['a', 'b', 'True a', 'True b'])
  plt.show()

main( )

#### end of file ####