#####
#   #
# 1 # THE DATA
#   #
#####
import os
import zipfile

###
#a# Unzip the dataset
###
datasets_path = '../../../datasets/ibean/'
datasets = [ 'ibeanstrain', 'ibeansval', 'ibeanstest' ]
for dataset in datasets:
    local_zip       = datasets_path + dataset + '.zip'
    zip_ref         = zipfile.ZipFile( local_zip, 'r' )
    output_path     = datasets_path + dataset
    zip_ref.extractall( output_path )
zip_ref.close()

###
#b# The data generators
###

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale               = 1./255,
      #rotation_range        = 40,
      #width_shift_range     = 0.2,
      #height_shift_range    = 0.2,
      #shear_range           = 0.2,
      #zoom_range            = 0.2,
      #horizontal_flip       = True,
      #fill_mode         = 'nearest'
      )

validation_datagen = ImageDataGenerator( rescale = 1./255 )

TRAIN_DIRECTORY_LOCATION    = datasets_path + 'ibeanstrain/train'
VAL_DIRECTORY_LOCATION      = datasets_path + 'ibeansval/validation'
TARGET_SIZE                 = ( 224, 224 )
CLASS_MODE                  = 'categorical'

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)

#####
#   #
# 2 # THE MODEL
#   #
#####
import tensorflow as tf

###
#a# The NN
###
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D( 16, ( 3, 3 ), activation = 'relu', input_shape = ( 224, 224, 3 ) ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The second convolution
    tf.keras.layers.Conv2D( 32, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The third convolution
    tf.keras.layers.Conv2D( 64, ( 3, 3 ), activation = 'relu' ),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # The fourth convolution
    tf.keras.layers.Conv2D( 128, ( 3, 3 ), activation = 'relu'),
    tf.keras.layers.MaxPooling2D( 2, 2 ),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten( ),
    # 512 neuron hidden layer
    tf.keras.layers.Dense( 512, activation = 'relu' ),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense( 3, activation = 'softmax' )
])

# This will print a summary of your model when you're done!
model.summary()

###
#b# The loss and optimizer.
###

LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = 'adam'

model.compile(
    loss = LOSS_FUNCTION,
    optimizer = OPTIMIZER,
    metrics = ['accuracy']
)

import argparse

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( '-e','--epochs', help = 'Epochs to train the model', default = 2, type = int, required = False )
args = parser.parse_args( )

EPOCHS      = args.epochs

history = model.fit(
      train_generator, 
      epochs = EPOCHS,
      verbose = 1,
      validation_data = validation_generator)

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,EPOCHS])
plt.ylim([0.4,1.0])
plt.show()
#### end of file ####
