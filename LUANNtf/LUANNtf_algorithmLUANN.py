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
debugCompareMultipleNetworksPerformanceGain = False

#select learningAlgorithm:
learningAlgorithmLIANN = False	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmNone = True	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only

#intialise network properties (configurable);	
useSparsity = True
if(useSparsity):
  sparsityProbabilityOfConnection = 0.1 #1-sparsity
supportSkipLayers = False #fully connected skip layer network
supportMultipleNetworks = False	#optional
if(debugCompareMultipleNetworksPerformanceGain):
	supportMultipleNetworks = True
	
#intialise network properties;
generateLargeNetwork = True	#required #CHECKTHIS: autoencoder does not require bottleneck	#for default LUANN operations
largeBatchSize = False	#not supported	#else train each layer using entire training set
generateNetworkStatic = False	#optional
generateDeepNetwork = True	#optional	#used for algorithm testing
if(generateDeepNetwork):
	generateNetworkStatic = True	#True: autoencoder requires significant number of neurons to retain performance?
shareComputationalUnits = False	
shareComputationalUnitsLayersExponentialDivergence = False
if(generateNetworkStatic):
	if(learningAlgorithmNone):	#only currently supported by learningAlgorithmNone
		shareComputationalUnits = True	#optional
		if(debugCompareMultipleNetworksPerformanceGain):
			shareComputationalUnits = False
		if(shareComputationalUnits):
			shareComputationalUnitsLayers = False
			shareComputationalUnitsNeurons = False	
			if(supportMultipleNetworks):
				shareComputationalUnitsLayers = True
				shareComputationalUnitsLayersExponentialDivergence = True	#simulate divergence of layers
				if(shareComputationalUnitsLayersExponentialDivergence):
					shareComputationalUnitsLayersDivergenceRate = 100	#muliplication of effective/unique networks per layer
			else:
				shareComputationalUnitsNeurons = True #prototype implementation for sharing computational units neurons in tensorflow (not required for smallDataset)	#shareComputationalUnitsNeurons are only possible because weights do not change	#reduces GPU RAM required to forward propagate large untrained net, but increases computational time (indexing of shared computational units)	#currently requires generateNetworkStatic (as each shared computational unit must have same number of inputs)
				#note shareComputationalUnitsNeurons:supportSkipLayers is supported, but will have to have enough GPU RAM to support Atrace/Ztrace for every unitxbatchSize in network
		
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
	if(useSparsity):
		generateLargeNetworkRatioMax = 1000
	else:
		generateLargeNetworkRatioMax = 100	#maximum number of neurons per layer required to provide significant performance
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

#Network parameters (predefinitions for supportMultipleNetworks);
numberOfLayers = 0
numberOfNetworks = 0
if(debugSingleLayerOnly):
	numberOfLayers = 1
else:
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2
numberOfNetworks = 0
if(supportMultipleNetworks):
	if(shareComputationalUnitsLayersExponentialDivergence):
		numberOfNetworks = pow(shareComputationalUnitsLayersDivergenceRate, numberOfLayers-1)
	else:
		if(generateLargeNetworkRatioMax == generateLargeNetworkRatio):
			#eg debugCompareMultipleNetworksPerformanceGain
			numberOfNetworks = 10	#100	
		else:
			numberOfNetworks = int(generateLargeNetworkRatioMax/generateLargeNetworkRatio) #normalise the number of networks based on the network layer size
			#numberOfNetworks = 10	#optional override

supportDimensionalityReductionLimitFrequency = False
supportDimensionalityReductionInhibitNeurons = False
if(supportDimensionalityReduction):
	
	supportDimensionalityReductionInhibitNeurons = True	#learn to inhibit neurons in net for a given task
	supportDimensionalityReductionMinimiseCorrelation = False #orig mode
	supportDimensionalityReductionRegulariseActivity = False	#reset neurons that are rarely used/fire (or are used/fire too often) across batches (batchSize >> numClasses) - indicates they do not contain useful information
	
	if(supportDimensionalityReductionMinimiseCorrelation):
		maxCorrelation = 0.95	#requires tuning
		supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	#if(supportDimensionalityReductionInhibitNeurons):
	
	if(supportDimensionalityReductionRegulariseActivity):
		supportDimensionalityReductionRegulariseActivityMinAvg = 0.01	#requires tuning
		supportDimensionalityReductionRegulariseActivityMaxAvg = 0.99	#requires tuning
		supportDimensionalityReductionRandomise	= True
	
	if(supportDimensionalityReductionInhibitNeurons):
		supportDimensionalityReductionFirstPhaseOnly = False
	else:
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
if(supportDimensionalityReductionInhibitNeurons):
  Nactive = {}  #effective bool [1.0 or 0.0]; whether neuron is active/inhibited

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

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	global randomUniformIndex
	if(shareComputationalUnits):
		if(shareComputationalUnitsNeurons):
			global WfSharedComputationalUnitsNeurons
			global BSharedComputationalUnitsNeurons
		if(shareComputationalUnitsLayers):
			global WfSharedComputationalUnitsLayers
			global BSharedComputationalUnitsLayers
	randomNormal = tf.initializers.RandomNormal()
	randomUniformIndex = tf.initializers.RandomUniform(minval=0.0, maxval=1.0)	#not available:	minval=0, maxval=numberOfSharedComputationalUnitsNeurons, dtype=tf.dtypes.int32; 

	if(shareComputationalUnits):
		shareComputationalUnitsChecks = False
		if(numberOfLayers >= 3):
			#shareComputationalUnitsNeurons must have at least 2 hidden layers
			if(n_h[1] == n_h[2]):
				shareComputationalUnitsChecks = True
			else:
				shareComputationalUnitsChecks = False
				print("shareComputationalUnitsNeurons must have at least 2 static sized hidden layers")
				exit()
		else:
			shareComputationalUnitsChecks = False
			print("shareComputationalUnitsNeurons must have at least 2 hidden layers")
			exit()
		if(shareComputationalUnitsChecks):
			if(shareComputationalUnitsLayers):
				WfSharedComputationalUnitsLayers = generateWeights([numberOfSharedComputationalUnitsLayers, n_h[1], n_h[1]])	#for every sharedComputationalUnitLayer, number of neurons on prior layer, number of neurons on current layer
				BSharedComputationalUnitsLayers = tf.zeros([numberOfSharedComputationalUnitsLayers, n_h[1]])
			elif(shareComputationalUnitsNeurons):		
				#current shareComputationalUnitsNeurons implementation requires enough GPU Ram to create and store a large numberOfSharedComputationalUnitsNeurons x staticHiddenLayerNumberNeurons weight array
				WfSharedComputationalUnitsNeurons = generateWeights([numberOfSharedComputationalUnitsNeurons, n_h[1]])	#for every sharedComputationalUnitNeuron, number of neurons on a prior layer
				BSharedComputationalUnitsNeurons = tf.zeros(numberOfSharedComputationalUnitsNeurons)
			
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
					Blayer = tf.zeros(n_h[l1])
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
			
			if(supportDimensionalityReductionInhibitNeurons):
				Nactivelayer = tf.ones(n_h[l1])
				Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = tf.Variable(Nactivelayer)
	
	if(supportMultipleNetworks):
		global WallNetworksFinalLayer
		global BallNetworksFinalLayer
		WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
		WallNetworksFinalLayer = tf.Variable(WlayerF)
		Blayer = tf.zeros(n_h[numberOfLayers])
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
	
#if(supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, y=None, networkIndex=1, l=None):
   return neuralNetworkPropagationLUANN(x, y=y, layer=l, networkIndex=networkIndex)
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred

#if(supportDimensionalityReduction):	
def neuralNetworkPropagationLUANNdimensionalityReduction(x, y=None, networkIndex=1):
	return neuralNetworkPropagationLUANN(x, y=y, layer=None, networkIndex=networkIndex, dimensionalityReduction=True)

def calculatePropagationLoss(x, y, networkIndex=1):
	costCrossEntropyWithLogits = False
	pred = neuralNetworkPropagation(x, networkIndex)
	target = y
	lossCurrent = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
	#acc = calculateAccuracy(pred, target)	#only valid for softmax class targets
	return lossCurrent

def neuralNetworkPropagationLUANN(x, y=None, layer=None, networkIndex=1, dimensionalityReduction=False):
	#y is only used by supportDimensionalityReductionInhibitNeurons

	pred = None 
	
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer

	if(layer is None):
		if(supportMultipleNetworks):
			maxLayer = numberOfLayers-1	 #ignore last layer (see neuralNetworkPropagationAllNetworksFinalLayer
		else:
			maxLayer = numberOfLayers
	else:
		maxLayer = layer
	#print("maxLayer = ", maxLayer)
	
	if(dimensionalityReduction):
		if(supportDimensionalityReductionInhibitNeurons):
			lossCurrent = calculatePropagationLoss(x, y, networkIndex)

	for l1 in range(1, maxLayer+1):	#ignore first layer
		
		#print("l1 = ", l1)
		
		A, Z = neuralNetworkPropagationLayerForward(l1, AprevLayer, networkIndex)
	
		if(dimensionalityReduction):
			if(l1 < numberOfLayers): #ignore last layer	#OLD: if(l1 < maxLayer):
				#print("dimensionalityReduction")
				if(supportDimensionalityReductionMinimiseCorrelation):
					ANNtf2_algorithmLIANN_math.neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)
				if(supportDimensionalityReductionRegulariseActivity):
					ANNtf2_algorithmLIANN_math.neuronActivationRegularisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, supportDimensionalityReductionRegulariseActivityMinAvg=supportDimensionalityReductionRegulariseActivityMinAvg, supportDimensionalityReductionRegulariseActivityMaxAvg=supportDimensionalityReductionRegulariseActivityMaxAvg)
				if(supportDimensionalityReductionInhibitNeurons):
					#randomly select a neuron k on layer to trial inhibition performance;
					Nactivelayer = Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")]
					NactivelayerBackup = Nactivelayer #tf.Variable(Nactivelayer)
					layerInhibitionIndex = tf.cast(randomUniformIndex([1])*n_h[l1], tf.int32)[0].numpy()
					print("layerInhibitionIndex = ", layerInhibitionIndex)
					Nactivelayer = tf.Variable(modifyTensorRowColumn(Nactivelayer, True, layerInhibitionIndex, 0.0, False))	#tf.Variable added to retain formatting
					#print("NactivelayerBackup = ", NactivelayerBackup)
					#print("Nactivelayer = ", Nactivelayer)
					Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = Nactivelayer
					loss = calculatePropagationLoss(x, y, networkIndex)
					#acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
					if(loss < lossCurrent):
						lossCurrent = loss
						print("loss < lossCurrent")
					else:
						Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")] = NactivelayerBackup
						print("loss !< lossCurrent")

		AprevLayer = A	#CHECKTHIS: note uses A value prior to weight updates
		if(recordNetworkTrace):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
		
		if(l1 == numberOfLayers):
			pred = tf.nn.softmax(Z)
		else:
			pred = A
			
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
					Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF), Blayer))	
			else:	
				WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
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
						Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndexl2, l2, "Atrace")], WlayerF), Blayer))	
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

		if(supportDimensionalityReductionInhibitNeurons):
			Z = tf.multiply(Z, Nactive[generateParameterNameNetwork(networkIndex, l1, "Nactive")])

		A = activationFunction(Z)
	
	return A, Z


def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
