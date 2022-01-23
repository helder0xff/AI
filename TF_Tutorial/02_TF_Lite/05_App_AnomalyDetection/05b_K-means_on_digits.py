# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-8-9-K-means.ipynb#scrollTo=yqxG6wjoR8LT

'''
Images here are 8x8 therefore 64 cells therefore 64 dimensions!!!!
'''

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

# CONSTANTS
# We will consider 0-1 to be abnormal and 2-9 to be normal.
NUM_CLUSTERS 			= 8
PERCENTILE_THRESHOLD 	= 99

def main():
	#####
	#	#
	# 1 # THE DATA
	#	#
	#####
	digits = load_digits()

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