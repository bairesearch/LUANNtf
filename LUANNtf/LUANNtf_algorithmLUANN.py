"""LUANNtf_algorithmLUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LUANNtf_main.py

# Usage:
see LUANNtf_main.py

# Description:
LUANNtf algorithm LUANN - define large untrained artificial neural network

"""

import tensorflow as tf
import numpy as np
import copy
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
#import LIANNtf_algorithmLIANN_math	#required for supportDimensionalityReduction
np.set_printoptions(suppress=True)

#debug parameters;
debugSmallNetwork = False	#small network for debugging matrix output
debugTrainAllLayers = False
debugFastTrain = False
debugPrintVerbose = False	#print weights and activations
debugSmallBatchSize = False	#small batch size for debugging matrix output
debugSingleLayerOnly = False
debugCompareMultipleNetworksPerformanceGain = False
debugLowEpochs = False

#select learningAlgorithm:
learningAlgorithmNone = True	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
#learningAlgorithmLIANN = False	#support disabled (see LIANNtf_algorithmLIANN)	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only	

#initialise network depth parameters;
generateDeepNetwork = False	#default: True	#optional	#used for algorithm testing

#initialise network size parameters;
generateLargeNetwork = False	#initialise (dependent var)
generateVeryLargeNetwork = False	#initialise (dependent var)
if(learningAlgorithmNone):
	#can pass different task datasets through a shared randomised net
	if(not debugSmallNetwork):
		#if(!supportMultipleNetworksSelect): large+ network is required
		generateLargeNetwork = False	#optional	default: False
		generateVeryLargeNetwork = False	#optional	default: True
		
#if(learningAlgorithmLIANN):
#	supportDimensionalityReduction = True	#mandatory	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN)

#initialise batch size parameters;
largeBatchSize = False	#True: train each layer using ~entire training set

#posthoc pruning customisation;
pruneConnections = False	#initialise (dependent var)
if(generateVeryLargeNetwork):
	pruneConnections = False	#optional	#prune final layer connections with low absolute value weights
	if(pruneConnections):
		pruneConnectionWeightThreshold = 1.0	#requires calibration
	
#LUANN core parameters (final layer training);
onlyTrainFinalLayer = True	#mandatory for LUANN
if(debugTrainAllLayers):
	onlyTrainFinalLayer = False

#intialise normalisation parameters	
normaliseFirstLayer = False	#default: False
if(onlyTrainFinalLayer):
	normaliseFirstLayer = True	#optimisation	#normalise input data based on mean/std

#intialise multiple network properties;	
supportMultipleNetworks = True	#optional
if(debugCompareMultipleNetworksPerformanceGain):
	supportMultipleNetworks = True
#if(supportMultipleNetworks):
supportMultipleNetworksSelect = True	#trial propagation performance of all networks, and select network that provides highest performance
supportMultipleNetworksMerge = False	#orig

#intialise experimental properties;	
supportMultipleNetworksStatic = False	#initialise (dependent var)
initialiseRandomBiases = False	#initialise (dependent var)
initialiseUniformWeights = False	#initialise (dependent var)
equaliseNumberExamplesPerClass = False	#initialise (dependent var)
if(supportMultipleNetworksSelect):
	supportMultipleNetworksStatic = True	#static number of networks / network size (not dynamically generated)
	initialiseRandomBiases = False	#trial
	initialiseUniformWeights = False	#trial
	equaliseNumberExamplesPerClass = False	#trial

#intialise sparsity properties;	
if(supportMultipleNetworksSelect):
	useSparsity = False
else:
	useSparsity = True
if(useSparsity):
  sparsityProbabilityOfConnection = 0.1 #1-sparsity

#intialise skip layer properties;	
supportSkipLayers = False #fully connected skip layer network

#shared computational units properties;	
generateNetworkStatic = False	#True: same number of neurons at each hidden layer (required for shareComputationalUnits)	#requires generateDeepNetwork
shareComputationalUnits = False	
shareComputationalUnitsLayersExponentialDivergence = False
if(generateDeepNetwork):
	if(generateNetworkStatic):
		if(learningAlgorithmNone):	#only currently supported by learningAlgorithmNone
			shareComputationalUnits = True	#optional
			if(debugCompareMultipleNetworksPerformanceGain):
				shareComputationalUnits = False
			if(shareComputationalUnits):
				shareComputationalUnitsLayers = False	#initialise (dependent var)	#requires at least 2 hidden layers
				shareComputationalUnitsNeurons = False	#initialise (dependent var)
				if(supportMultipleNetworks):
					shareComputationalUnitsLayers = True	#default: True
					shareComputationalUnitsNeurons = False	#default: False
					if(not supportMultipleNetworksStatic):
						shareComputationalUnitsLayersExponentialDivergence = True	#simulate divergence of layers
						if(shareComputationalUnitsLayersExponentialDivergence):
							shareComputationalUnitsLayersDivergenceRate = 100	#muliplication of effective/unique networks per layer
				else:
					shareComputationalUnitsNeurons = True	#default: True #prototype implementation for sharing computational units neurons in tensorflow (not required for smallDataset)	#shareComputationalUnitsNeurons are only possible because weights do not change	#reduces GPU RAM required to forward propagate large untrained net, but increases computational time (indexing of shared computational units)	#currently requires generateNetworkStatic (as each shared computational unit must have same number of inputs)
					#note shareComputationalUnitsNeurons:supportSkipLayers is supported, but will have to have enough GPU RAM to support Atrace/Ztrace for every unitxbatchSize in network

#initialise network size parameters (detailed);
if(not supportMultipleNetworksStatic):
	if(useSparsity):
		generateLargeNetworkRatioMax = 1000
	else:
		generateLargeNetworkRatioMax = 100	#maximum number of neurons per layer required to provide significant performance
if(generateVeryLargeNetwork):
	if(supportMultipleNetworksStatic):
		if(useSparsity):
			generateLargeNetworkRatio = 100	#orig: 1000
		else:
			generateLargeNetworkRatio = 10
	else:
		if(supportMultipleNetworks):
			if(shareComputationalUnits and shareComputationalUnitsLayersExponentialDivergence):
				generateLargeNetworkRatio = 1
			else:
				if(debugCompareMultipleNetworksPerformanceGain):
					generateLargeNetworkRatio = generateLargeNetworkRatioMax
				else:
					generateLargeNetworkRatio = 10	#default: 10
		else:
			generateLargeNetworkRatio = generateLargeNetworkRatioMax
else:
	if(generateLargeNetwork):
		generateLargeNetworkRatio = 3
	else:
		generateLargeNetworkRatio = 1

#initialise network depth parameters (detailed);
numberOfLayers = 0
numberOfNetworks = 0
if(debugSingleLayerOnly):
	numberOfLayers = 1
else:
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2

#intialise multiple network properties (detailed);	
numberOfNetworks = 0
if(supportMultipleNetworks):
	if(supportMultipleNetworksStatic):
		numberOfNetworks = 1000	#100	#100	#2	#10	#1000
	else:
		if(shareComputationalUnitsLayersExponentialDivergence):
			numberOfNetworks = pow(shareComputationalUnitsLayersDivergenceRate, numberOfLayers-1)
		else:
			if(generateLargeNetworkRatioMax == generateLargeNetworkRatio):
				#eg debugCompareMultipleNetworksPerformanceGain
				numberOfNetworks = 10	#100	
			else:
				numberOfNetworks = int(generateLargeNetworkRatioMax/generateLargeNetworkRatio) #normalise the number of networks based on the network layer size

#learningAlgorithmLIANN support currently disabled (see LIANNtf_algorithmLIANN)
#if(supportDimensionalityReduction):
#	#supportDimensionalityReductionAlgorithmX = False
#	
#	supportDimensionalityReductionFirstPhaseOnly = False	#perform LIANN in first phase only (x epochs of training), then apply hebbian learning at final layer
#	supportDimensionalityReductionLimitFrequency = False
#	if(supportDimensionalityReductionLimitFrequency):
#		supportDimensionalityReductionLimitFrequencyStep = 1000	

		
#network/activation parameters;
#forward excitatory connections;
Wf = {}
B = {}
if(supportMultipleNetworksMerge):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
recordNetworkTrace = False
if(supportSkipLayers or shareComputationalUnitsLayersExponentialDivergence):
	recordNetworkTrace = True
	Ztrace = {}
	Atrace = {}
if(shareComputationalUnits):
	#shared computational units are only currently used for connections between static sized layers (ie not input and output layers):
	if(shareComputationalUnitsLayers):
		numberOfSharedComputationalUnitsLayers = 1000	#number of unique shared computational unit layers (this should be less than the total number of layers in a multi-network of static sized layers)
		WfSharedComputationalUnitsLayers = None
		WfIndex = {}
		BSharedComputationalUnitsLayers = None
		BIndex = {}		
	elif(shareComputationalUnitsNeurons):
		numberOfSharedComputationalUnitsNeurons = 1000	#number of unique shared computational unit neurons (this should be less than the total number of units in the network static sized layers)
		WfSharedComputationalUnitsNeurons = None
		WfIndex = {}
		BSharedComputationalUnitsNeurons = None
		BIndex = {}
		#WfSharedComputationalUnitsSubnets	#not currently implemented
#if(inhibitionAlgorithmBinary):
#	Nactive = {}  #effective bool [1.0 or 0.0]; whether neuron is active/inhibited

#Network parameters;
n_h = []
#numberOfLayers = 0
#numberOfNetworks = 0
datasetNumClasses = 0

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
			learningRate = 0.005	#0.05
		else:
			batchSize = 100	#3	#100
			learningRate = 0.005
	if(generateDeepNetwork):
		if(debugLowEpochs):
			numEpochs = 10
		else:
			numEpochs = 100	#higher num epochs required for convergence
	else:
		numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = 100
	else:
		trainingSteps = 1000	#10000	#1000
	
	print("batchSize = ", batchSize)
	
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	
def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global datasetNumClasses

	firstHiddenLayerNumberNeurons = num_input_neurons*generateLargeNetworkRatio
			
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
			
	return numberOfLayers
	
def generateWeights(shape):
	initialisedWeights = randomNormal(shape) 
	if(useSparsity):
		sparsityMatrixMask = tf.random.uniform(shape, minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
		sparsityMatrixMask = tf.math.less(sparsityMatrixMask, sparsityProbabilityOfConnection)
		sparsityMatrixMask = tf.cast(sparsityMatrixMask, dtype=tf.dtypes.float32)
		initialisedWeights = tf.multiply(initialisedWeights, sparsityMatrixMask)	
	return initialisedWeights

def generateBiases(shape):
	if(initialiseRandomBiases):
		initialisedBiases = randomNormal(shape)
	else:
		initialisedBiases = tf.zeros(shape)
	return initialisedBiases

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	global randomUniformIndex
	global randomUniform
	if(shareComputationalUnits):
		if(shareComputationalUnitsNeurons):
			global WfSharedComputationalUnitsNeurons
			global BSharedComputationalUnitsNeurons
		if(shareComputationalUnitsLayers):
			global WfSharedComputationalUnitsLayers
			global BSharedComputationalUnitsLayers
	randomNormal = tf.initializers.RandomNormal()
	randomUniformIndex = tf.initializers.RandomUniform(minval=0.0, maxval=1.0)	#not available:	minval=0, maxval=numberOfSharedComputationalUnitsNeurons, dtype=tf.dtypes.int32; 
	randomUniform = tf.initializers.RandomUniform(minval=-1.0, maxval=1.0)
	
	if(shareComputationalUnits):
		shareComputationalUnitsChecks = False
		if(numberOfLayers >= 3):
			#shareComputationalUnits must have at least 2 hidden layers
			if(n_h[1] == n_h[2]):
				shareComputationalUnitsChecks = True
			else:
				shareComputationalUnitsChecks = False
				print("shareComputationalUnits must have at least 2 static sized hidden layers")
				exit()
		else:
			shareComputationalUnitsChecks = False
			print("shareComputationalUnits must have at least 2 hidden layers")
			exit()
		if(shareComputationalUnitsChecks):
			if(shareComputationalUnitsLayers):
				WfSharedComputationalUnitsLayers = generateWeights([numberOfSharedComputationalUnitsLayers, n_h[1], n_h[1]])	#for every sharedComputationalUnitLayer, number of neurons on prior layer, number of neurons on current layer
				BSharedComputationalUnitsLayers = generateBiases([numberOfSharedComputationalUnitsLayers, n_h[1]])
			elif(shareComputationalUnitsNeurons):		
				#current shareComputationalUnitsNeurons implementation requires enough GPU Ram to create and store a large numberOfSharedComputationalUnitsNeurons x staticHiddenLayerNumberNeurons weight array
				WfSharedComputationalUnitsNeurons = generateWeights([numberOfSharedComputationalUnitsNeurons, n_h[1]])	#for every sharedComputationalUnitNeuron, number of neurons on a prior layer
				BSharedComputationalUnitsNeurons = generateBiases([numberOfSharedComputationalUnitsNeurons])
			
	for networkIndex in range(1, numberOfNetworks+1):
		#print("networkIndex = ", networkIndex)
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
				
			useSameBranchNetworkValues = False
			if(shareComputationalUnitsLayersExponentialDivergence):
				if(l1 < numberOfLayers): #ignore last layer
					useSameBranchNetworkValues, networkIndexCurrentBranchStart = calculateUseSameBranchNetworkValues(l1, networkIndex)
					#if(useSameBranchNetworkValues):
					#	print("\t\tuseSameBranchNetworkValues = ", useSameBranchNetworkValues)
					#else:
					#	print("\t\tuseSameBranchNetworkValues = ", useSameBranchNetworkValues)

			if(not useSameBranchNetworkValues):
				#print("\tl1 = ", l1)
				useUniqueLayerWeights = True
				if(shareComputationalUnits):
					if((l1 > 1) and (l1 < numberOfLayers)):
						#shared computational units are only currently used for connections between static sized layers (ie not input and output layers)
						useUniqueLayerWeights = False
				if(useUniqueLayerWeights):
					if(supportSkipLayers):
						for l2 in range(0, l1):
							if(l2 < l1):
								WlayerF = generateWeights([n_h[l2], n_h[l1]]) 
								Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(WlayerF)
					else:
						WlayerF = generateWeights([n_h[l1-1], n_h[l1]]) 
						Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(WlayerF)
					Blayer = generateBiases([n_h[l1]])
					B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(Blayer)
				else:
					if(supportSkipLayers):
						for l2 in range(1, l1):	#shareComputationalUnitsNeurons does not currently support skip layer connections to input layer (which has different layer size)
							if(l2 < l1):
								if(shareComputationalUnitsLayers):
									WlayerFIndex = tf.squeeze(tf.cast(randomUniformIndex([1])*numberOfSharedComputationalUnitsLayers, tf.int32))
								elif(shareComputationalUnitsNeurons):	
									WlayerFIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
								WfIndex[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "WfIndex")] = tf.Variable(WlayerFIndex)
					else:
						if(shareComputationalUnitsLayers):
							WlayerFIndex = tf.squeeze(tf.cast(randomUniformIndex([1])*numberOfSharedComputationalUnitsLayers, tf.int32))
						elif(shareComputationalUnitsNeurons):	
							WlayerFIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
						WfIndex[generateParameterNameNetwork(networkIndex, l1, "WfIndex")] = WlayerFIndex
					if(shareComputationalUnitsLayers):
						BlayerIndex = tf.squeeze(tf.cast(randomUniformIndex([1])*numberOfSharedComputationalUnitsLayers, tf.int32))
					elif(shareComputationalUnitsNeurons):	
						BlayerIndex = tf.cast(randomUniformIndex([n_h[l1]])*numberOfSharedComputationalUnitsNeurons, tf.int32)
					BIndex[generateParameterNameNetwork(networkIndex, l1, "BIndex")] = tf.Variable(BlayerIndex)		
			
			if(recordNetworkTrace):
				if(not useSameBranchNetworkValues):
					Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
					Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))				
			
			#if(inhibitionAlgorithmBinary):
			#	Nactivelayer = tf.ones(n_h[l1])
			#	Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = tf.Variable(Nactivelayer)
	
	if(supportMultipleNetworksMerge):
		global WallNetworksFinalLayer
		global BallNetworksFinalLayer
		WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
		WallNetworksFinalLayer = tf.Variable(WlayerF)
		Blayer = generateBiases([n_h[numberOfLayers]])
		BallNetworksFinalLayer	= tf.Variable(Blayer)	#not currently used

def calculateUseSameBranchNetworkValues(l1, networkIndex):
	useSameBranchNetworkValues = False
	networkIndexCurrentBranchStart = 0
	if(shareComputationalUnitsLayersExponentialDivergence):
		numberOfUniqueNetworkBranchesAtLayer = pow(shareComputationalUnitsLayersDivergenceRate, l1)
		networkBranchRedundantSizeAtLayer = numberOfNetworks//numberOfUniqueNetworkBranchesAtLayer
		#print("shareComputationalUnitsLayersDivergenceRate = ", shareComputationalUnitsLayersDivergenceRate)
		#print("numberOfUniqueNetworkBranchesAtLayer = ", numberOfUniqueNetworkBranchesAtLayer)
		#print("networkBranchRedundantSizeAtLayer = ", networkBranchRedundantSizeAtLayer)
		#print("networkIndex = ", networkIndex)
		#print("l1 = ", l1)
		#remainder = (networkIndex-1)%networkBranchRedundantSizeAtLayer
		#print("remainder = ", remainder)
		if((networkIndex-1)%networkBranchRedundantSizeAtLayer == 0):
			#print("useSameBranchNetworkValues = False")
			useSameBranchNetworkValues = False
			networkIndexCurrentBranchStart = networkIndex
		else:
			#print("useSameBranchNetworkValues = True")
			useSameBranchNetworkValues = True
			#print("networkBranchRedundantSizeAtLayer = ", networkBranchRedundantSizeAtLayer)
			networkIndexCurrentBranchStart = networkIndex - ((networkIndex-1)%networkBranchRedundantSizeAtLayer)	#((networkIndex-1)//networkBranchRedundantSizeAtLayer) + 1
	return useSameBranchNetworkValues, networkIndexCurrentBranchStart
									
def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf2)
	return neuralNetworkPropagationLUANNallLayers(x, networkIndex=networkIndex)
	#return neuralNetworkPropagationLUANNtest(x, networkIndex=1)
def neuralNetworkPropagationLUANNallLayers(x, networkIndex=1):
	return neuralNetworkPropagationLUANN(x, networkIndex=networkIndex)
	
#if(supportMultipleNetworksMerge):
def neuralNetworkPropagationLayer(x, y=None, networkIndex=1, l=None):
   return neuralNetworkPropagationLUANN(x, y=y, layer=l, networkIndex=networkIndex)
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred

#if(supportDimensionalityReduction):	
#def neuralNetworkPropagationLUANNdimensionalityReduction(x, y=None, networkIndex=1):
#	return neuralNetworkPropagationLUANN(x, y=y, layer=None, networkIndex=networkIndex, dimensionalityReduction=True)

def calculatePropagationLoss(x, y, networkIndex=1):
	costCrossEntropyWithLogits = False
	pred = neuralNetworkPropagation(x, networkIndex)
	target = y
	lossCurrent = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
	#acc = calculateAccuracy(pred, target)	#only valid for softmax class targets
	return lossCurrent

def neuralNetworkPropagationLUANN(x, y=None, layer=None, networkIndex=1, dimensionalityReduction=False):
	#y is only used by inhibitionAlgorithmBinary

	pred = None 
	
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer

	if(layer is None):
		if(supportMultipleNetworksMerge):
			maxLayer = numberOfLayers-1	 #ignore last layer (see neuralNetworkPropagationAllNetworksFinalLayer
		else:
			maxLayer = numberOfLayers
	else:
		maxLayer = layer
	#print("maxLayer = ", maxLayer)
	
	#moved 15 Mar 2022
	#if(dimensionalityReduction):
	#	if(inhibitionAlgorithmBinary):
	#		lossCurrent = calculatePropagationLoss(x, y, networkIndex)

	for l1 in range(1, maxLayer+1):	#ignore first layer
		
		#print("l1 = ", l1)
		
		A, Z = neuralNetworkPropagationLayerForward(l1, AprevLayer, networkIndex)
	
		#learningAlgorithmLIANN support currently disabled (see LIANNtf_algorithmLIANN)
		#if(dimensionalityReduction):
		#	if(l1 < numberOfLayers): #ignore last layer	#OLD: if(l1 < maxLayer):
		#		#print("dimensionalityReduction")
		#		if(supportDimensionalityReductionAlgorithmX):
					
		AprevLayer = A	#CHECKTHIS: note uses A value prior to weight updates
		if(recordNetworkTrace):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A

		if(onlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)
						
		if(l1 == numberOfLayers):
			pred = tf.nn.softmax(Z)
			#print("pred = ", pred)
		else:
			pred = A

		if(debugPrintVerbose):
			print("pred = ", pred)
						
	return pred

def neuralNetworkPropagationLayerForward(l1, AprevLayer, networkIndex=1):
	
	A, Z = (None, None)
	
	useSameBranchNetworkValues = False
	if(shareComputationalUnitsLayersExponentialDivergence):
		if(l1 < numberOfLayers): #ignore last layer
			useSameBranchNetworkValues, networkIndexCurrentBranchStart = calculateUseSameBranchNetworkValues(l1, networkIndex)
			if(l1 > 1):	#ignore first layer
				lPrev = l1-1
				useSameBranchNetworkValuesPrev, networkIndexCurrentBranchStartPrev = calculateUseSameBranchNetworkValues(lPrev, networkIndex)
				AprevLayer = Atrace[generateParameterNameNetwork(networkIndexCurrentBranchStartPrev, lPrev, "Atrace")]
				#print("networkIndex = ", networkIndex)
				#print("useSameBranchNetworkValuesPrev = ", useSameBranchNetworkValuesPrev)
				#print("networkIndexCurrentBranchStartPrev = ", networkIndexCurrentBranchStartPrev)
				#print("AprevLayer = ", AprevLayer)

	#print("AprevLayer = ", AprevLayer)
	if(not useSameBranchNetworkValues):
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
					Z = tf.add(Z, tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF))	
				Z = tf.add(Z, Blayer)
			else:	
				WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
				if(debugPrintVerbose):
					print("WlayerF = ", WlayerF)
				Z = tf.add(tf.matmul(AprevLayer, WlayerF), Blayer)
		else:
			#dynamically generate layer biases;
			BlayerIndex = BIndex[generateParameterNameNetwork(networkIndex, l1, "BIndex")]
			#Blayer = BSharedComputationalUnitsNeurons[BlayerIndex]
			if(shareComputationalUnitsLayers):
				Blayer = tf.gather(BSharedComputationalUnitsLayers, BlayerIndex)
			elif(shareComputationalUnitsNeurons):
				Blayer = tf.gather(BSharedComputationalUnitsNeurons, BlayerIndex)
			#print("Blayer = ", Blayer)	
			if(supportSkipLayers):
				Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
				for l2 in range(1, l1):	#shareComputationalUnitsNeurons does not currently support skip layer connections to input layer (which has different layer size)
					if(l2 < l1):
						#dynamically generate layer weights;
						networkIndexl2 = networkIndex
						if(shareComputationalUnitsLayersExponentialDivergence):
							useSameBranchNetworkValues2, networkIndexCurrentBranchStart2 = calculateUseSameBranchNetworkValues(l2, networkIndex)	
							if(useSameBranchNetworkValues2):
								networkIndexl2 = networkIndexCurrentBranchStart2
									
						WlayerFIndex = WfIndex[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "WfIndex")]
						#WlayerF = WfSharedComputationalUnitsNeurons[WlayerFIndex]
						if(shareComputationalUnitsLayers):
							WlayerF = tf.gather(WfSharedComputationalUnitsLayers, WlayerFIndex)
						elif(shareComputationalUnitsNeurons):
							WlayerF = tf.gather(WfSharedComputationalUnitsNeurons, WlayerFIndex)
						Z = tf.add(Z, tf.matmul(Atrace[generateParameterNameNetwork(networkIndexl2, l2, "Atrace")], WlayerF))	
				Z = tf.add(Z, Blayer)
			else:
				#dynamically generate layer weights;
				WlayerFIndex = WfIndex[generateParameterNameNetwork(networkIndex, l1, "WfIndex")]
				#WlayerF = WfSharedComputationalUnitsNeurons[WlayerFIndex]
				if(shareComputationalUnitsLayers):
					WlayerF = tf.gather(WfSharedComputationalUnitsLayers, WlayerFIndex)
				elif(shareComputationalUnitsNeurons):
					WlayerF = tf.gather(WfSharedComputationalUnitsNeurons, WlayerFIndex)
				#print("WlayerF = ", WlayerF)
				#print("Blayer = ", Blayer)
				Z = tf.add(tf.matmul(AprevLayer, WlayerF), Blayer)

		#if(inhibitionAlgorithmBinary):
		#	Z = tf.multiply(Z, Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")])

		A = activationFunction(Z)
	
	return A, Z


def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A

#pruneConnections:
def pruneFinalLayerWeights():
	if(supportMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	for networkIndex in range(1, maxNetwork+1):
		l1 = numberOfLayers
		if(supportSkipLayers):
			for l2 in range(0, l1):
				if(l2 < l1):
					l1size = n_h[l1]
					l2size = n_h[l2]
					weights = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
					weights = pruneWeight(weights, pruneConnectionWeightThreshold)
					Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = weights
		else:
			l2 = l1-1
			l1size = n_h[l1]
			l2size = n_h[l2]
			weights = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
			print("pre weights = ", weights)
			weights = pruneWeight(weights, pruneConnectionWeightThreshold)
			print("post weights = ", weights)
			Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = weights
				
def pruneWeight(weights, threshold):
	weights = tf.where(tf.logical_or(tf.math.greater(weights, threshold), tf.math.less(weights, -threshold)), weights, 0.0) 
	return weights
