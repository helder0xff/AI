# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-8-13-Autoencoders.ipynb

import matplotlib.pyplot as plt
from mechanize import History
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#####
#	#
# 0 # THE MODEL
#	#
#####
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")]) # Smallest Layer Defined Here
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#####
#	#
# x # HELPER FUNCTIONS
#	#
#####
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold), loss

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

def main():
    #####
	#	#
	# 1 # THE DATA
	#	#
	#####
    # Download the dataset
    dataframe = pd.read_csv('../../../datasets/ecg.csv', header=None)
    raw_data = dataframe.values

    # Get data ready for training.
    # The last element contains the labels
    labels = raw_data[:, -1]

    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )    

    # Normalize the data.
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    # Separate normal (labeled as 1) from abnormal
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    #####
	#	#
	# 2 # THE AUTO-ENCODER
	#	#
	#####    
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mae')
    history = autoencoder.fit(  normal_train_data, 
                                normal_train_data,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_data),
                                shuffle=True)

    #####
	#	#
	# 3 # SOME NICE PRINTING TO SHOW WHAT IS HAPPENING
	#	#
	#####
    encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    plt.plot(normal_test_data[0],'b')
    plt.plot(decoded_imgs[0],'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral' )
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.title("Normal hearthbeat")
    plt.show()

    encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    plt.plot(anomalous_test_data[0],'b')
    plt.plot(decoded_imgs[0],'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='lightcoral' )
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.title("Abnormal hearthbeat")
    plt.show()

    #####
	#	#
	# 4 # ANOMALY DETECTION
	#	#
	#####
    # Calculate the train loss for each data.
    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

    # Choose a threshold based on the mean and the standard deviation.
    threshold = np.mean(train_loss) + np.std(train_loss)

    preds, scores = predict(autoencoder, test_data, threshold)
    print_stats(preds, test_labels)

main()

#### end of file ####
