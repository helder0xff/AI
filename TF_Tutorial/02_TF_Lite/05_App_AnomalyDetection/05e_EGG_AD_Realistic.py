# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-8-16-Assignment.ipynb#scrollTo=YfIk2es3hJEd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# CONSTANTS
EMBEDDING_SIZE  = 2
THRESHOLD       = 0.4 # Chosenf from the ROC

#####
#	#
# 0 # THE MODEL
#	#
#####
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      layers.Dense(EMBEDDING_SIZE, activation="relu")]) # Smallest Layer Defined Here
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
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

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    # Mix some anomalies within the normal data (more realistic case)
    portion_of_anomaly_in_training = 0.1 #10% of training data will be anomalies
    end_size = int(len(normal_train_data)/(10-portion_of_anomaly_in_training*10))
    combined_train_data = np.append(normal_train_data, anomalous_test_data[:end_size], axis=0)

    #####
	#	#
	# 2 # THE AUTO-ENCODER
	#	#
	#####    
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mae')
    history = autoencoder.fit(  combined_train_data, 
                                combined_train_data,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_data),
                                shuffle=True)

    #####
	#	#
	# 4 # ANOMALY DETECTION
	#	#
	#####
    # Calculate the train loss for each data.
    reconstructions = autoencoder.predict(test_data)
    train_loss = tf.keras.losses.mae(reconstructions, test_data)

    # Choose a threshold based on the mean and the standard deviation.
    threshold = THRESHOLD

    preds, scores = predict(autoencoder, test_data, threshold)
    print_stats(preds, test_labels)

    #####
	#	#
	# 3 # PRINT ROC
	#	#
	#####   
    reconstructions = autoencoder(test_data)
    loss = tf.keras.losses.mae(reconstructions, test_data)
    fpr = []
    tpr = []
    #the test labels are flipped to match how the roc_curve function expects them.
    flipped_labels = 1-test_labels 
    fpr, tpr, thresholds = roc_curve(flipped_labels, loss)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    # plot some thresholds
    thresholds_every=20
    thresholdsLength = len(thresholds)
    colorMap=plt.get_cmap('jet', thresholdsLength)
    for i in range(0, thresholdsLength, thresholds_every):
        threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
        plt.scatter(fpr[i], tpr[i], c='black')
        plt.text(fpr[i] - 0.03, tpr[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15})

    plt.show()

main()
