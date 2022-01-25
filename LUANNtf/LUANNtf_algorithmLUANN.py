"""LUANNtf_algorithmLUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LUANNtf_main.py

# Usage:
see LUANNtf_main.py

# Description:
LUANN algorithm LUANN - define large untrained artificial neural network

"""

import tensorflow as tf
import numpy as np
import copy
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import ANNtf2_algorithmLIANN_math	#required for supportDimensionalityReduction
np.set_printoptions(suppress=True)

#debug parameters;
debugSmallNetwork = False	#not supported #small network for debugging matrix output
debugSmallBatchSize = False	#not supported #small batch size for debugging matrix output
debugSingleLayerOnly = False
debugFastTrain = False	#not supported

#select learningAlgorithm:
learningAlgorithmLIANN = False	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmNone = True	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only

#intialise network properties (configurable);	
supportSkipLayers = False #fully connected skip layer network

supportMultipleNetworks = True	#optional (required to activate set trainMultipleNetworks=True in LUANNtf_main)


#intialise network properties;
generateLargeNetwork = True	#required #CHECKTHIS: autoencoder does not require bottleneck	#for default LUANN operations
largeBatchSize = False	#not supported	#else train each layer using entire training set
generateNetworkStatic = False	#optional
generateDeepNetwork = True	#optional	#used for algorithm testing
if(generateDeepNetwork):
	generateNetworkStatic = True	#True: autoencoder requires significant number of neurons to retain performance?

#learning algorithm customisation;
generateVeryLargeNetwork = False
if(learningAlgorithmLIANN):
	if(not debugSmallNetwork):
		generateVeryLargeNetwork = True	#default: True
	supportDimensionalityReduction = True	#mandatory	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN)
elif(learningAlgorithmNone):
	#can pass different task datasets through a shared randomised net
	generateVeryLargeNetwork = True
	supportDimensionalityReduction = False

if(generateVeryLargeNetwork):
	generateLargeNetworkRatio = 100	#100	#default: 10
else:
	if(generateLargeNetwork):
		generateLargeNetworkRatio = 3
	else:
		generateLargeNetworkRatio = 1

supportDimensionalityReductionLimitFrequency = False
if(supportDimensionalityReduction):
	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	maxCorrelation = 0.95	#requires tuning
	supportDimensionalityReductionLimitFrequency = True
	if(supportDimensionalityReductionLimitFrequency):
		supportDimensionalityReductionLimitFrequencyStep = 1000


#network/activation parameters;
#forward excitatory connections;
Wf = {}
Wb = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global batchSize
	
	if(debugSmallBatchSize):
		batchSize = 10
		learningRate = 0.001
	else:
		if(largeBatchSize):
			batchSize = 1000	#current implementation: batch size should contain all examples in training set
			learningRate = 0.05
		else:
			batchSize = 100	#3	#100
			learningRate = 0.005
	if(generateDeepNetwork):
		numEpochs = 100	#higher num epochs required for convergence
	else:
		numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
		
	firstHiddenLayerNumberNeurons = num_input_neurons*generateLargeNetworkRatio
	if(debugSingleLayerOnly):
		numberOfLayers = 1
	else:
		if(generateDeepNetwork):
			numberOfLayers = 3
		else:
			numberOfLayers = 2
			
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
			
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						WlayerF = randomNormal([n_h[l2], n_h[l1]]) 
						Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(WlayerF)
			else:
				WlayerF = randomNormal([n_h[l1-1], n_h[l1]]) 
				Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(WlayerF)
				
			Blayer = tf.zeros(n_h[l1])
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(Blayer)

			if(supportSkipLayers):
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
	
	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			Blayer = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer	= tf.Variable(Blayer)	#not currently used
			
def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf2)
	return neuralNetworkPropagationLUANNfinalLayer(x, networkIndex=networkIndex)
	#return neuralNetworkPropagationLUANNtest(x, networkIndex=1)
def neuralNetworkPropagationLUANNfinalLayer(x, networkIndex=1):
	return neuralNetworkPropagationLUANN(x, layer=numberOfLayers, networkIndex=networkIndex)
	
#if(supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
   return neuralNetworkPropagationLUANN(x, layer=l, networkIndex=networkIndex)
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred

#if(supportDimensionalityReduction):	
def neuralNetworkPropagationLUANNdimensionalityReduction(x, networkIndex=1):
	return neuralNetworkPropagationLUANN(x, layer=None, networkIndex=networkIndex, dimensionalityReduction=True)

def neuralNetworkPropagationLUANN(x, layer=None, networkIndex=1, dimensionalityReduction=False):

	pred = None 
	
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer

	if(layer is None):
		maxLayer = numberOfLayers
	else:
		maxLayer = layer
			
	for l1 in range(1, maxLayer+1):	#ignore first layer
		
		A, Z = neuralNetworkPropagationLayerForward(l1, AprevLayer, networkIndex)
	
		if(dimensionalityReduction):
			if(l1 < maxLayer): #ignore last layer
				#print("dimensionalityReduction")
				ANNtf2_algorithmLIANN_math.neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=Wb, Wbname="Wb", updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)
		
		AprevLayer = A	#CHECKTHIS: note uses A value prior to weight updates
		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
		
		if(l1 == numberOfLayers):
			pred = tf.nn.softmax(Z)
		else:
			pred = A
			
	return pred

def neuralNetworkPropagationLayerForward(l1, AprevLayer, networkIndex=1):
	
	if(supportSkipLayers):
		Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
		for l2 in range(0, l1):
			WlayerF = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
			Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")]))	
	else:	
		WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
		Z = tf.add(tf.matmul(AprevLayer, WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")])
	A = activationFunction(Z)
	
	return A, Z


def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
