# For Numpy
import matplotlib.pyplot as plt
import numpy as np
import pprint
import re
import sys
# For TensorFlow Lite (also uses some of the above)
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
from tensorflow import keras
import pathlib
import pprint
import re
import sys

def quantizeAndReconstruct(weights):
    """
    @param W: np.ndarray

    This function computes the scale value to map fp32 values to int8. The function returns
    a weight matrix in fp32, that is representable using 8-bits.
    """

    # Compute the range of the weight.
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    range = max_weight - min_weight
        
    # Compute the scale
    max_int8 = 2**8
    scale = range / max_int8

    # Compute the midpoint
    midpoint = np.mean([max_weight, min_weight])

    # Next, we need to map the real fp32 values to the integers. For this, we make use of the computed scale. By diving the weight 
    # matrix with the scale, the weight matrix has a range between (-128, 127). Now, we can simply round the full precision numbers
    # to the closest integers.
    centered_weights = weights - midpoint
    quantized_weights = np.rint(centered_weights / scale)

    # Now, we can reconstruct the values back to fp32.
    reconstr_weights = scale * quantized_weights + midpoint
    return reconstr_weights

def getMaxQuantizationError(original_weights, reconstr_weights):
	errors = reconstr_weights - original_weights
	max_error = np.max(errors)
	
	return max_error

def main():
	############################################################################
	# GET A BUNCH OF WEIGHTS
	original_weights = np.random.randn(256, 256)
	############################################################################

	############################################################################
	# QUANTIZE AND RECONSTRUCT WEIGHTS
	reconstr_weights = quantizeAndReconstruct(original_weights)

	# We can use np.unique to check the number of unique floating point numbers in the weight matrix.
	uniques = np.unique(reconstr_weights).shape[0]
	assert uniques <= 2**8

	print("\n############################################################")
	print("QUANTIZE AND RECONSTRUCT WEIGHTS")
	print("Max Error:", getMaxQuantizationError(original_weights, reconstr_weights))
	print("Unique Values:", uniques)
	print("############################################################\n")
	############################################################################ 
	
main()

#### end of file ####