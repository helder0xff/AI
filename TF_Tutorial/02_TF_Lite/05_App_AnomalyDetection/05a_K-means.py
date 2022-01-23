# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-8-9-K-means.ipynb#scrollTo=yqxG6wjoR8LT

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
NUM_CENTERS 			= 3
NUM_SAMPLES 			= 3000
SAMPLES_10_PERCENT 		= int(NUM_SAMPLES / 10)
NUM_ANOMALY_SAMPLES 	= SAMPLES_10_PERCENT
NUM_ANOMALY_CENTRES 	= 2
PERCENTILE_THRESHOLD	= 99

def main():
	#####
	#	#
	# 1 # CREATE NOT-ANOMALY SAMPLES
	#	#
	#####
	# Create some NUM_CENTERS centered data 
	samples, labels  = make_blobs(		n_samples 	= NUM_SAMPLES, 
										centers 	= NUM_CENTERS,
	                                   	cluster_std=0.40, 
	                                   	random_state=0)

	# Plot it
	plt.scatter(samples[:, 0], samples[:, 1], s=50)
	plt.title("Cluster of Samples")
	plt.show()

	#####
	#	#
	# 2 # K-MEAN PREDICTION
	#	#
	#####
	# K-Means prediciton
	keep_predicting = True

	while(True == keep_predicting):
		kmeans = KMeans(n_clusters = NUM_CENTERS) 					# Init KMeans instance.
		kmeans.fit(samples[ : NUM_SAMPLES - SAMPLES_10_PERCENT]) 					# Fit with the samples
		predictions = kmeans.predict(samples[NUM_SAMPLES - SAMPLES_10_PERCENT : ])	# Predict

		# You might seen that in the output the labels and predictions do not correspond. That is wrong.
		# The algorithm works perfectly but NEEDS MAPPING. What make_blobs label as 1, Kmeans might label
		# it differently.
		#print(labels[NUM_SAMPLES - SAMPLES_10_PERCENT : ])
		#print(predictions)

		correct_cnt = 0
		for i in range(len(predictions)):
			if predictions[i] == labels[ NUM_SAMPLES - SAMPLES_10_PERCENT + i]:
				correct_cnt += 1
		accuracy = int((correct_cnt / SAMPLES_10_PERCENT) * 100)
		if 90 < accuracy:
			keep_predicting = False


	# Plot the predictions
	plt.scatter(samples[NUM_SAMPLES - SAMPLES_10_PERCENT : , 0], samples[NUM_SAMPLES - SAMPLES_10_PERCENT : , 1], c=predictions, s=50, cmap='viridis')
	centers = kmeans.cluster_centers_ # The centers.
	plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
	plt.title("K-means labelling.")
	plt.show()

	#####
	#	#
	# 3 # CREATE ANOMALY SAMPLES
	#	#
	#####
	# Create some NUM_CENTERS centered data 
	anomaly_samples, anomaly_labels  = make_blobs(	n_samples 		= NUM_ANOMALY_SAMPLES, 
													centers 		= NUM_ANOMALY_CENTRES,
	                                				cluster_std 	= 0.40, 
	                                				random_state	= 1)

	plt.scatter(samples[:, 0], samples[:, 1], s=50);
	plt.scatter(anomaly_samples[:,0], anomaly_samples[:,1], s=50)
	plt.title("Added Anomaly Samples")
	plt.show()

	#####
	#	#
	# 4 # CALCULATE PERCENTILE DISTANCE
	#	#
	#####
	# Calculate distance of each sample to each centre.
	samples_distances = kmeans.transform(samples)

	# Group distances of each sample of a cluster.
	center_distances = {key: [] for key in range(NUM_CENTERS)}
	for i in range(len(labels)):
		min_distance = samples_distances[i][labels[i]]
		center_distances[labels[i]].append(min_distance)

	# Calculate the
	percentile_distance = {key: np.percentile(	center_distances[key], 	\
	                                          	PERCENTILE_THRESHOLD)   \
	                                			for key in center_distances.keys()}

	fig, ax = plt.subplots()

	colors = []
	for i in range(len(samples)):
		min_distance = samples_distances[i][labels[i]]
		if (min_distance > percentile_distance[labels[i]]):
			colors.append(4)
		else:
			colors.append(labels[i])


	ax.scatter(samples[:, 0], samples[:, 1], c=colors, s=50, cmap='viridis')

	for i in range(len(centers)):
		circle = plt.Circle((centers[i][0], centers[i][1]),percentile_distance[i], color='black', alpha=0.1);
		ax.add_artist(circle)
	plt.title("Percentile Distance")
	plt.show()

	#####
	#	#
	# 5 # PLOT ANOMALY SAMPLES AND CALCULATE FALSE + AND -
	#	#
	#####
	fig, ax = plt.subplots()

	anomaly_distances = kmeans.transform(anomaly_samples)
	anomaly_prediction = kmeans.predict(anomaly_samples)

	#combine all the data
	combined_distances = [*samples_distances, *anomaly_distances]
	combined_labels = [*labels, *anomaly_prediction]
	all_samples = np.array([*samples, *anomaly_samples])

	false_neg=0
	false_pos=0

	colors = []
	for i in range(len(all_samples)):
		min_distance = combined_distances[i][combined_labels[i]]
		if (min_distance > percentile_distance[combined_labels[i]]):
			colors.append(4)
			if (i<NUM_SAMPLES):
				false_pos+=1
		else:
			colors.append(combined_labels[i])
			if (i>=NUM_SAMPLES):
				false_neg+=1

	ax.scatter(all_samples[:, 0], all_samples[:, 1], c=colors, s=50, cmap='viridis')

	for i in range(len(centers)):
		circle = plt.Circle((centers[i][0], centers[i][1]),percentile_distance[i], color='black', alpha=0.1);
		ax.add_artist(circle)

	plt.title("Prediction")
	plt.show()

	print('Normal datapoints misclassified as abnormal: ', false_pos * 100 / NUM_SAMPLES)
	print('Abnormal datapoints misclassified as normal: ', false_neg * 100 / NUM_ANOMALY_SAMPLES)	

main()

#### end of life ####