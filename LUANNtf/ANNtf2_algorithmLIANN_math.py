"""ANNtf2_algorithmLIANN_math.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN math (PCA simulation, correlation matrix, etc)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters

np.set_printoptions(linewidth=np.inf)

#only required by ANNtf2_algorithmLIANN_math:SVD/PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

SVDmatrixTypeExamplesByNeurons = True
SVDmatrixTypeSpikeCoincidence = False

	
#uses np not tf;
def generateSVDinputMatrix(l, n_h, AprevLayer):
	if(SVDmatrixTypeExamplesByNeurons):
		SVDinputMatrix = AprevLayer.numpy()
	elif(SVDmatrixTypeSpikeCoincidence):
		SVDinputMatrix = generateSpikeCoincidenceMatrix(l, n_h, AprevLayer)
	return SVDinputMatrix
	
def generateSpikeCoincidenceMatrix(l, n_h, AprevLayer):
	#creates co-occurence count matrix of dimensionality (num_neurons, num_neurons)
	layerSize = n_h[l-1]	#or AprevLayer.shape[1]
	batchSize = AprevLayer.shape[0]
	spikeCoincidenceMatrix = np.empty([layerSize, layerSize])
	AprevLayerNP = AprevLayer.numpy()	#perform operations in numpy to save dev time
	for k1 in range(layerSize):
		for k2 in range(layerSize):
			spikeCoincidences = np.logical_and(AprevLayerNP[:,k1], AprevLayerNP[:,k2])
			spikeCoincidenceMatrix[k1, k2] = np.sum(spikeCoincidences)
	print("spikeCoincidenceMatrix = ", spikeCoincidenceMatrix)
	return spikeCoincidenceMatrix

def calculateSVD(M, k):		
	#print("M = ", M)		
	U, Sigma, VT = randomized_svd(M, n_components=k, n_iter=5, random_state=None)	#n_iter = ?
	return U, Sigma, VT

def calculateWeights(l, n_h, SVDinputMatrix, U, Sigma, VT):
	#incomplete
	#SVDinputMatrix: batchSize*prevLayerSize
	#U: batchSize*layerSize
	SVDinputMatrixT = np.transpose(SVDinputMatrix)
	layerSize = n_h[l]
	prevLayerSize = n_h[l-1]
	AWnumpy = np.matmul(SVDinputMatrixT, U)  	#incomplete: some function of SVDinputMatrix, U, Sigma, VT	#dimensions: prevLayerSize*layerSize
	AW = tf.convert_to_tensor(AWnumpy)
	return AW

	
def calculateCorrelationMean(A):
	correlationsOffDiagonal = calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=1.0, getOffDiagonalCorrelationMatrix=False)
	meanCorrelation = calculateCorrelationMatrixOffDiagonalsMean(correlationsOffDiagonal)
	#print("meanCorrelation = ", meanCorrelation)
	
	return meanCorrelation
	
	#alternate methods;
	#normalisedStddev = tf.math.reduce_std(A, axis=1)/tf.math.reduce_mean(A, axis=1)	#batched	similarity(A) = stddev(A)/avg(A)
	#mean = tf.math.reduce_mean(A, axis=1)	#batched
	#mean = tf.expand_dims(mean, axis=1)
	#differenceFromMean(tf.subtract(A, mean))
	
def calculateCorrelationMatrixOffDiagonalsMean(correlationsOffDiagonal):
	#alternate methods:
	correlationsOffDiagonal = np.abs(correlationsOffDiagonal)	#account for negative correlations
	meanCorrelation = np.mean(correlationsOffDiagonal)

	#this method assumes bivariate normality - CHECKTHIS assumption is not violated
	#https://www.researchgate.net/post/average_of_Pearson_correlation_coefficient_values
		#https://stats.stackexchange.com/questions/109028/fishers-z-transform-in-python
	#fishersZ = np.arctanh(correlations)
	#fishersZaverage = np.mean(fishersZ)
	#inverseFishersZ = np.tanh(fishersZaverage)
	#meanCorrelation = inverseFishersZ
	return meanCorrelation
	
def calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=1.0, getOffDiagonalCorrelationMatrix=True):
	Anumpy = A.numpy()
	correlationMatrixX = np.transpose(Anumpy)	#2-D array containing multiple variables and observations. Each row of x represents a variable, and each column a single observation of all those variables. Also see rowvar below.	#https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
	correlationMatrix = np.corrcoef(correlationMatrixX)	#create correlation matrix across all k neuron dimensions (for every sample instance in the batch)

	if(getOffDiagonalCorrelationMatrix):
		diagonalOffset = 0	#exclude diagonal matrix entries
	else:
		diagonalOffset = 1	#include diagonal matrix entries - CHECKTHIS
		
	offdiagonalIndices = np.triu_indices_from(correlationMatrix, diagonalOffset)	#get off diagonal correlations	#https://stackoverflow.com/questions/14129979/mean-of-a-correlation-matrix-pandas-data-fram
	correlationMatrix[np.isnan(correlationMatrix)] = nanReplacementValue	#set all correlationMatrix nan entries as 1.0 (high correlation), to disincentivise all values of a neuron k being the same value across the batch
	correlationsOffDiagonal = correlationMatrix[offdiagonalIndices[0], offdiagonalIndices[1]]
	
	correlationMatrixOffDiagonal = np.copy(correlationMatrix)
	correlationMatrixOffDiagonal[offdiagonalIndices[0], offdiagonalIndices[1]] = 0.0
	
	if(getOffDiagonalCorrelationMatrix):
		return correlationMatrixOffDiagonal
	else:
		return correlationsOffDiagonal
	
