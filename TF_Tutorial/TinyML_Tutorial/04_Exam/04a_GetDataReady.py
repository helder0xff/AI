"""
This script split the flowers dataset in the three different channels RGB in 
order to experiment with its merging and check the different permances.
The obtained sets from this script will be used in 04b_* script.
"""

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from itertools import chain, combinations
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser(description='Command line for the experiments.')
parser.add_argument( 	'-s',
						'--source',
						 help = 'soruce from where to import dataset', 
						 type = str,
						 choices = ['originalCap', 'original'],
						 required = True )
args = parser.parse_args( )
SOURCE = '/' + args.source

def getMinimunResolutionOfShapeSets( shapesSet ):
	"""
	Get minimun resolution on a set of shapes.
	"""
	min_shape = ''
	minResolution = sys.maxsize
	for shape in shapesSet:
		resolution = shape[ 0 ] * shape[ 1 ]
		if resolution < minResolution:
			minResolution = resolution
			min_shape = shape
	if len(min_shape) == 3:
		min_shape = list(min_shape)
		min_shape.pop(-1)
		min_shape = tuple(min_shape)
	
	return min_shape

def splitChannels( image ):
	"""
	Split the channels
	"""
	height, width, channels = image.shape

	splittedImage = [ ]
	for channel in range( channels ):
		splittedImage.append( [] )
		channelArray = splittedImage[ channel ]
		for h in range( height ):
			channelArray.append( [ ] )
			for w in range( width ):
				channelArray[ h ].append( image[ h ][ w ][ channel ] )

	return np.asarray( splittedImage, dtype = np.uint8)	

def adjustChannelsDimension( channels ):
	"""
	Adjust the dimension of the channels to have all of them the same one.
	The smallest is chosen.
	"""
	shapesSet = set()
	for channel in channels:		
		shapesSet.add( channel.shape )
	dim = getMinimunResolutionOfShapeSets( shapesSet )
	resizedChannels = [ ]
	resizedChannel = ''
	for channel in channels:
		resizedChannel = cv2.resize( channel, dim)
		resizedChannels.append( resizedChannel )

	return (resizedChannels, dim)

def mergeChannels( channels ):
	"""
	Merge all the channels in the same array.
	"""
	resizedChannels, dim = adjustChannelsDimension( channels )
	height 	= dim[0]
	width 	= dim[1]

	mergedImage = [ ]
	for w in range( width ):
		row = [ ]
		for h in range( height ):
			pixel = [ ]
			for channel in resizedChannels:
				pixel.append( channel[ w ][ h ] )
			row.append( pixel )
		mergedImage.append( row )

	return np.asarray( mergedImage, dtype = np.uint8 )

def main( ):
	# Set paths.
	root_path 	= '../../../datasets/flower_photos'
	sub_sets = [ '/training', '/test', '/validation' ]
	source_set = SOURCE
	flowers = [ '/daisy', '/dandelion', '/roses', '/sunflowers', '/tulips' ]	
	channels = [ 'R', 'G', 'B' ]
	# Create a combinational list of channels of the form: ['R', 'G', 'B', 'RG', 'RB', 'GB', 'RGB']
	channelsCombination = chain.from_iterable(combinations(channels, r) for r in range(len(channels)+1))
	channelsCombination = list(map(''.join, channelsCombination))[1:]

	# FIND THE SMALLEST RESOLUTION
	# Search through all the images on the dataset to find the one with 
	# the lowest resolution. This dimension will be used in the training script
	# to be able to match all the images and the NN imput layer.
	print("Finding smallest resolution...")
	shapes_set = set()
	totalCnt = 0
	for sub_set in sub_sets:
		for flower in flowers:
			inputpath = root_path + sub_set + source_set + flower + '/'
			progress = 0		
			for file in os.listdir( inputpath ):
				image = cv2.imread( inputpath + file )
				shapes_set.add( image.shape )
				totalCnt += 1
				print( str(image.shape), end='\r' )
	min_dim = getMinimunResolutionOfShapeSets(shapes_set)
	print("SMALLEST RESOLUTION:", min_dim, '(use it in the NN training script)')	

	# GET READY
	# Delete existing splitted channels dataset.
	# (Not necessary once we see everything is done properly and the splitted
	# channels dataset is done.)
	for sub_set in sub_sets:		
		basePath = root_path + sub_set + '/'
		for combination in channelsCombination:	
			path = basePath + combination
			try:
				shutil.rmtree(path)
			except:
				pass

	# BUILD DATASET
	print("Building splitted channels datasets...")
	progress = 1
	for sub_set in sub_sets:		
		for flower in flowers:
			inputpath = root_path + sub_set + source_set + flower + '/'
			localCnt = 0		
			for file in os.listdir( inputpath ):
				# Some activity printing.
				print( progress,'/',totalCnt, end='\r' )
				progress += 1
				localCnt += 1

				# OPEN IMAGE
				image = cv2.imread( inputpath + file )

				'''
				# In case you want to see the splitting and mergin process, not recomended.
				plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
				print(inputpath+file)
				plt.title('original')
				plt.show()
				'''

				# SPLIT CHANNELS
				red, green, blue = splitChannels( image )

				# RESIZE CHANNELS
				# This is done for the sake of complexity. In the real system
				# we will have different channels with different resolutions.
				# So here we slightly increase and decrease resolution of
				# certain channels.
				dim = min_dim								
				red = cv2.resize(red, dim)

				dim = tuple(int(e*0.9) for e in min_dim)
				green = cv2.resize(green, dim)

				dim = tuple(int(e*1.1) for e in min_dim)
				blue = cv2.resize(blue, dim)

				# MERGE CHANNELS				
				for combination in channelsCombination:
					imageChannels = []
					if 'R' in combination:
						imageChannels.append( red )
					if 'G' in combination:
						imageChannels.append( green )
					if 'B' in combination:
						imageChannels.append( blue )
					image_merged = mergeChannels( imageChannels )

					'''
					# In case you want to see the splitting and mergin process, not recomended.
					if(2 != len(combination)):
						if( 3 == len(combination)):
							plt.imshow(cv2.cvtColor(image_merged, cv2.COLOR_BGR2RGB), interpolation = 'nearest')
						else:
							plt.imshow(image_merged, interpolation='nearest')							
						plt.title(combination)
						plt.show()					
					'''

					# SAVE IMAGE
					outputpath = 	root_path + 	\
									sub_set + '/' + 	\
									combination + 		\
									flower + '/' +		\
									str( localCnt ) + '.npy'
					Path(root_path + sub_set + '/' + combination + flower ).mkdir(parents=True, exist_ok=True)
					Path(outputpath).touch()									
					np.save( outputpath,  image_merged )
	print("...done.")				

main( )

#### end of file ####
