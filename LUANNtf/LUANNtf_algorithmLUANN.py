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
learningAlgorithmLIANN = True	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmNone = False	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only

#intialise network properties (configurable);	
supportSkipLayers = False #fully connected skip layer network

supportMultipleNetworks = True	#optional (required to activate set trainMultipleNetworks=True in LUANNtf_main)


#intialise network properties;
generateLargeNetwork = True	#required #CHECKTHIS: autoencoder does not require bottleneck	#for default LUANN operations
largeBatchSize = False	#not supported	#else train each layer using entire training set
generateNetworkStatic = False	#optional
generateDeepNetwork = True	#optional	#used for algorithm testing
shareComputationalUnits = False
if(generateDeepNetwork):
	generateNetworkStatic = True	#True: autoencoder requires significant number of neurons to retain performance?
if(generateNetworkStatic):
	if(learningAlgorithmNone):	#only currently supported by learningAlgorithmNone
		shareComputationalUnits = True #prototype implementation for sharing computational units (neurons/subnets) in tensorflow (not required for smallDataset)	#shareComputationalUnits are only possible because weights do not change	#reduces GPU RAM required to forward propagate large untrained net, but increases computational time (indexing of shared computational units)	#currently requires generateNetworkStatic (as each shared computational unit must have same number of inputs)
		#note shareComputationalUnits:supportSkipLayers is supported, but will have to have enough GPU RAM to support Atrace/Ztrace for every unitxbatchSize in network
	
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
	supportDimensionalityReductionRegulariseActivity = True	#reset neurons that are rarely used/fire (or are used/fire too often) across batches - indicates they do not contain useful information
	if(supportDimensionalityReductionRegulariseActivity):
		supportDimensionalityReductionRegulariseActivityMinAvg = 0.1
		supportDimensionalityReductionRegulariseActivityMaxAvg = 0.9
	supportDimensionalityReductionFirstPhaseOnly = True	#perform LIANN in first phase only (x epochs of training), then apply hebbian learning at final layer
	if(supportDimensionalityReductionFirstPhaseOnly):
		supportDimensionalityReductionLimitFrequency = False
		supportDimensionalityReductionFirstPhaseOnlyNumEpochs = 1
	else:
		supportDimensionalityReductionLimitFrequency = True
		if(supportDimensionalityReductionLimitFrequency):
			supportDimensionalityReductionLimitFrequencyStep = 1000
			
#network/activation parameters;
#forward excitatory connections;
Wf = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}
if(shareComputationalUnits):
	#shared computational units are only currently used for connections between static sized layers (ie not input and output layers):
	numberOfSharedComputationalUnitsNeurons = 1000	#number of unique shared computational units/neurons (this should be less than the total number of units in the network static sized layers)
	WfSharedComputationalUnitsNeurons = None
	WfIndex = {}
	BSharedComputationalUnitsNeurons = None
	BIndex = {}
	#WfSharedComputationalUnitsSubnets	#not currently implemented
	

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
	global shareComputationalUnits
	if(shareComputationalUnits):
		global WfSharedComputationalUnitsNeurons
		global BSharedComputationalUnitsNeurons
	randomNormal = tf.initializers.RandomNormal()
	
	if(shareComputationalUnits):
		randomUniformIndex = tf.initializers.RandomUniform(minval=0.0, maxval=1.0)	#not available:  minval=0, maxval=numberOfSharedComputationalUnitsNeurons, dtype=tf.dtypes.int32; 
		shareComputationalUnitsChecks = False
		if(numberOfLayers >= 3):
			#shareComputationalUnits must have at least 2 hidden layers
			if(n_h[1] == n_h[2]):
				#current shareComputationalUnits implementation requires enough GPU Ram to create and store a large numberOfSharedComputationalUnitsNeurons x staticHiddenLayerNumberNeurons weight array
				WfSharedComputationalUnitsNeurons = randomNormal([numberOfSharedComputationalUnitsNeurons, n_h[1]])	#for every sharedComputationalUnit, number of neurons on a prior layer
				BSharedComputationalUnitsNeurons = tf.zeros(numberOfSharedComputationalUnitsNeurons)
			else:
				shareComputationalUnits = False
				print("shareComputationalUnits must have at least 2 static sized hidden layers")
				exit()
		else:
			shareComputationalUnits = False
			print("shareComputationalUnits must have at least 2 hidden layers")
			exit()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
			
			useUniqueLayerWeights = True
			if(shareComputationalUnits):
				if((l1 > 1) and (l1 < numberOfLayers)):
					#shared computational units are only currently used for connections between static sized layers (ie not input and output layers)
					useUniqueLayerWeights = False
			
			if(useUniqueLayerWeights):
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
			else:
				if(supportSkipLayers):
					for l2 in range(1, l1):	#shareComputationalUnits does not currently support skip layer connections to input layer (which has different layer size)
						if(l2 < l1):
							WlayerFIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
							#print("WlayerFIndex = ", WlayerFIndex)
							WfIndex[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "WfIndex")] = tf.Variable(WlayerFIndex)
				else:
					WlayerFIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
					#print("WlayerFIndex = ", WlayerFIndex)
					WfIndex[generateParameterNameNetwork(networkIndex, l1, "WfIndex")] = WlayerFIndex
				BlayerIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
				BIndex[generateParameterNameNetwork(networkIndex, l1, "BIndex")] = tf.Variable(BlayerIndex)		
			
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
				ANNtf2_algorithmLIANN_math.neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)
				if(supportDimensionalityReductionRegulariseActivity):
					ANNtf2_algorithmLIANN_math.neuronActivationRegularisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, supportDimensionalityReductionRegulariseActivityMinAvg=supportDimensionalityReductionRegulariseActivityMinAvg, supportDimensionalityReductionRegulariseActivityMaxAvg=supportDimensionalityReductionRegulariseActivityMaxAvg)
					
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

	useUniqueLayerWeights = True
	if(shareComputationalUnits):
		if((l1 > 1) and (l1 < numberOfLayers)):
			#shared computational units are only currently used for connections between static sized layers (ie not input and output layers)
			useUniqueLayerWeights = False
					
	if(useUniqueLayerWeights):	
		Blayer = B[generateParameterNameNetwork(networkIndex, l1, "B")]
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				WlayerF = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
				Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF), Blayer))	
		else:	
			WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
			Z = tf.add(tf.matmul(AprevLayer, WlayerF), Blayer)
	else:
		#dynamically generate layer biases;
		BlayerIndex = BIndex[generateParameterNameNetwork(networkIndex, l1, "BIndex")]
		#Blayer = BSharedComputationalUnitsNeurons[BlayerIndex]
		Blayer = tf.gather(BSharedComputationalUnitsNeurons, BlayerIndex)
		#print("Blayer = ", Blayer)	
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(1, l1):	#shareComputationalUnits does not currently support skip layer connections to input layer (which has different layer size)
				if(l2 < l1):
					#dynamically generate layer weights;
					WlayerFIndex = WfIndex[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "WfIndex")]
					#WlayerF = WfSharedComputationalUnitsNeurons[WlayerFIndex]
					WlayerF = tf.gather(WfSharedComputationalUnitsNeurons, WlayerFIndex)
					Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF), Blayer))	
		else:
			#dynamically generate layer weights;
			WlayerFIndex = WfIndex[generateParameterNameNetwork(networkIndex, l1, "WfIndex")]
			#WlayerF = WfSharedComputationalUnitsNeurons[WlayerFIndex]
			WlayerF = tf.gather(WfSharedComputationalUnitsNeurons, WlayerFIndex)
			Z = tf.add(tf.matmul(AprevLayer, WlayerF), Blayer)
			
	A = activationFunction(Z)
	
	return A, Z


def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
