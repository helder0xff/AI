# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-8-9-K-means.ipynb#scrollTo=yqxG6wjoR8LT

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# CONSTANTS
DIMENSIONS 				= 2
NUM_CLUSTERS 			= 8
PERCENTILE_THRESHOLD 	= 99

def main():
	#####
	#	#
	# 1 # THE DATA
	#	#
	#####
	# Load it
	digits = load_digits()

	# Project it
	tsne = TSNE(n_components = DIMENSIONS, init = 'random', random_state = 0)
	digits_proj = tsne.fit_transform(digits.data)

	print('Original Dimension', digits.data.shape[1])
	print('Projected Dimension', digits_proj.shape[1])

	#Visualize our new data
	fig, ax = plt.subplots()
	ax.scatter(digits_proj[:, 0], digits_proj[:, 1],c=digits.target, s=50, cmap='viridis')
	plt.title("Projected Data")
	plt.show()

	# SEPARATE THE DATA
	normal_data = []
	abnormal_data = []

	normal_label = []
	abnormal_label = []
	
	#separate our data arbitrarily into normal (2-9) and abnormal (0-1)
	for i in range(len(digits.target)):
		if digits.target[i]< 10 - NUM_CLUSTERS:
			abnormal_data.append(digits.data[i])
			abnormal_label.append(digits.target[i])
		else:
			normal_data.append(digits.data[i])
			normal_label.append(digits.target[i])

	#####
	#	#
	# 2 # CLUSTER IT
	#	#
	#####
	# Cluster
	kmeans = KMeans(n_clusters = NUM_CLUSTERS, random_state = 0)
	kmeans.fit(normal_data)

	# Get Percentile
	normal_predictions 	= kmeans.predict(normal_data)
	normal_distances 	= kmeans.transform(normal_data)
	center_distances = {key: [] for key in range(NUM_CLUSTERS)}
	for i in range(len(normal_predictions)):
		min_distance = normal_distances[i][normal_predictions[i]]
		center_distances[normal_predictions[i]].append(min_distance)

	percentile_distances = {key: np.percentile(	center_distances[key], \
												PERCENTILE_THRESHOLD)   \
	                                			for key in center_distances.keys()}		

	#####
	#	#
	# 3 # PREDICT ANOMALIES
	#	#
	#####
	abnormal_predictions 	= kmeans.predict(abnormal_data)
	abnormal_distances 		= kmeans.transform(abnormal_data)

	#combine all the data
	combined_distances = [*normal_distances, *abnormal_distances]
	combined_predictions = [*normal_predictions, *abnormal_predictions]

	normal_data_length = len(normal_data)
	all_data = np.array([*normal_data, *abnormal_data])

	false_neg=0
	false_pos=0

	for i in range(len(all_data)):
		min_distance = combined_distances[i][combined_predictions[i]]
		if (min_distance > percentile_distances[combined_predictions[i]]):
			if (i<normal_data_length): #training data is first
				false_pos+=1
	else:
		if (i>=normal_data_length):
			false_neg+=1

	print('Normal datapoints misclassified as abnormal: ', false_pos)
	print('Abnormal datapoints misclassified as normal: ', false_neg)

main()

#### end of file ####
