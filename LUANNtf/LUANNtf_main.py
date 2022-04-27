"""LUANNtf_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
	LUANNsequentialInput only:
conda install nltk
conda install spacy
python3 -m spacy download en_core_web_md

# Usage:
python3 LUANNtf_main.py

# Description:
LUANNtf - train last layer of a large untrained artificial neural network (LUANN)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

from itertools import zip_longest
from ANNtf2_operations import *
import random
import ANNtf2_loadDataset


costCrossEntropyWithLogits = False
import LUANNtf_algorithmLUANN as LUANNtf_algorithm

suppressGradientDoNotExistForVariablesWarnings = True

useSmallSentenceLengths = False

trainMultipleFiles = False
if(LUANNtf_algorithm.supportMultipleNetworks):
	trainMultipleNetworks = True
else:
	trainMultipleNetworks = False
if(trainMultipleNetworks):
	#numberOfNetworks = 10
	numberOfNetworks = LUANNtf_algorithm.numberOfNetworks
	if(numberOfNetworks == 1):	#train at least 2 networks (required for tensorflow code execution consistency)
		trainMultipleNetworks = False
else:
	numberOfNetworks = 1


if(trainMultipleFiles):
	randomiseFileIndexParse = True
	fileIndexFirst = 0
	fileIndexLast = batchSize-1
	#if(useSmallSentenceLengths):
	#	fileIndexLast = 11
	#else:
	#	fileIndexLast = 1202	#defined by wiki database extraction size
else:
	randomiseFileIndexParse = False
			
dataset = "SmallDataset"

if(dataset == "SmallDataset"):
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
	#trainMultipleFiles = False	#required
	smallDatasetDefinitionsHeader = {'index':0, 'name':1, 'fileName':2, 'classColumnFirst':3}	
	smallDatasetDefinitions = [
	(0, "New Thyroid", "new-thyroid.data", True),
	(1, "Swedish Auto Insurance", "UNAVAILABLE.txt", False),	#AutoInsurSweden.txt BAD
	(2, "Wine Quality Dataset", "winequality-whiteFormatted.csv", False),
	(3, "Pima Indians Diabetes Dataset", "pima-indians-diabetes.csv", False),
	(4, "Sonar Dataset", "sonar.all-data", False),
	(5, "Banknote Dataset", "data_banknote_authentication.txt", False),
	(6, "Iris Flowers Dataset", "iris.data", False),
	(7, "Abalone Dataset", "UNAVAILABLE", False),	#abaloneFormatted.data BAD
	(8, "Ionosphere Dataset", "ionosphere.data", False),
	(9, "Wheat Seeds Dataset", "seeds_datasetFormatted.txt", False),
	(10, "Boston House Price Dataset", "UNAVAILABLE", False)	#housingFormatted.data BAD
	]
	dataset2FileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("dataset2FileName = ", dataset2FileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
	
debugUseSmallSequentialInputDataset = False
if(debugUseSmallSequentialInputDataset):
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset4FileNameStart = "Xdataset4Part"
xmlDatasetFileNameEnd = ".xml"



def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	return LUANNtf_algorithm.defineTrainingParameters(dataset)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return LUANNtf_algorithm.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks)	
	
def defineNeuralNetworkParameters():
	return LUANNtf_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return LUANNtf_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	return LUANNtf_algorithm.neuralNetworkPropagation(x, networkIndex)

#if(ANNtf2_algorithm.supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex, l):
	return LUANNtf_algorithm.neuralNetworkPropagationLayer(x, networkIndex, l)
def neuralNetworkPropagationAllNetworksFinalLayer(x):
	return LUANNtf_algorithm.neuralNetworkPropagationAllNetworksFinalLayer(x)
	
def trainBatch(e, batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display):
	
	executeFinalLayerHebbianLearning = True
	#if(LUANNtf_algorithm.supportDimensionalityReduction):
	#	if(LUANNtf_algorithm.supportDimensionalityReductionFirstPhaseOnly):
	#		if(e < LUANNtf_algorithm.supportDimensionalityReductionFirstPhaseOnlyNumEpochs):
	#			executeFinalLayerHebbianLearning = False

	#print("trainMultipleFiles error: does not support greedy training for LUANN")
	if(executeFinalLayerHebbianLearning):
		#print("executeFinalLayerHebbianLearning")
		if(trainMultipleNetworks):
			#LUANNtf_algorithm.neuralNetworkPropagationLUANNallLayers(batchX, networkIndex)	#propagate without performing final layer optimisation	#why executed?
			pass
		else:
			loss, acc = executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
	
	#if(LUANNtf_algorithm.supportDimensionalityReduction):
	#	executeLIANN = False
	#	if(LUANNtf_algorithm.supportDimensionalityReductionFirstPhaseOnly):
	#		if(e < LUANNtf_algorithm.supportDimensionalityReductionFirstPhaseOnlyNumEpochs):
	#			executeLIANN = True			
	#	elif(LUANNtf_algorithm.supportDimensionalityReductionLimitFrequency):
	#		if(batchIndex % LUANNtf_algorithm.supportDimensionalityReductionLimitFrequencyStep == 0):
	#			executeLIANN = True
	#	else:
	#		executeLIANN = True
	#	if(executeLIANN):
	#		#print("executeLIANN")
	#		LUANNtf_algorithm.neuralNetworkPropagationLUANNdimensionalityReduction(batchX, batchY, networkIndex)	

	pred = None
	if(display):
		if(not trainMultipleNetworks):
			loss, acc = calculatePropagationLoss(batchX, batchY, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex)	#display final layer loss
			print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
			
def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex)

	l1 = numberOfLayers	#LUANN only ever perfoms weight optimisation of final layer
	
	#must syncronise with defineNeuralNetworkParameters;
	#train specific layer weights;
	Wlist = []
	Blist = []
	if(LUANNtf_algorithm.supportSkipLayers):
		for l2 in range(0, l1):
			if(l2 < l1):
				Wlist.append(LUANNtf_algorithm.Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")])
	else:
		Wlist.append(LUANNtf_algorithm.Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")])
	Blist.append(LUANNtf_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])
	trainableVariables = Wlist + Blist
	WlistLength = len(Wlist)
	BlistLength = len(Blist)

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
		
	return loss, acc
					
def calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1):
	acc = 0	#only valid for softmax class targets 

	pred = LUANNtf_algorithm.neuralNetworkPropagationLUANNallLayers(x, networkIndex)
	target = y 
	#print("pred.shape = ", pred.shape)
	#print("target.shape = ", target.shape)
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
	#print("target = ", target)
	#print("pred = ", pred)
	#print("x = ", x)
	#print("y = ", y)
	#print("2 loss = ", loss)
	#print("2 acc = ", acc)

			
	return loss, acc



#if(ANNtf2_algorithm.supportMultipleNetworks):

def testBatchAllNetworksFinalLayer(batchX, batchY, datasetNumClasses, numberOfLayers):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	
	pred = neuralNetworkPropagationAllNetworksFinalLayer(AfinalHiddenLayerTensor)
	acc = calculateAccuracy(pred, batchY)
	print("Combined network: Test Accuracy: %f" % (acc))
	
	
	
def trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display):
	
	AfinalHiddenLayerList = []
	#print("numberOfNetworks = ", numberOfNetworks)
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)	
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor.shape = ", AfinalHiddenLayerTensor.shape)
	
	executeOptimisationAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, optimizer)

	pred = None
	if(display):
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, costCrossEntropyWithLogits)
		print("Combined network: batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))
						
def executeOptimisationAllNetworksFinalLayer(x, y, datasetNumClasses, optimizer):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	Wlist = []
	Blist = []
	Wlist.append(LUANNtf_algorithm.WallNetworksFinalLayer)
	Blist.append(LUANNtf_algorithm.BallNetworksFinalLayer)
	trainableVariables = Wlist + Blist

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
			
def calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits):
	acc = 0	#only valid for softmax class targets 
	pred = neuralNetworkPropagationAllNetworksFinalLayer(x)
	#print("calculatePropagationLossAllNetworksFinalLayer: pred = ", pred)
	target = y
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 

	return loss, acc
	
	
		
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, LUANNsequentialInputTypesMaxLength, useSmallSentenceLengths, LUANNsequentialInputTypeTrainWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp

def processDatasetLUANN(LUANNsequentialInputTypeIndex, inputVectors):

	percentageDatasetTrain = ANNtf2_loadDataset.percentageDatasetTrain
	
	datasetNumExamples = inputVectors.shape[0]

	datasetNumExamplesTrain = int(float(datasetNumExamples)*percentageDatasetTrain/100.0)
	datasetNumExamplesTest = int(float(datasetNumExamples)*(100.0-percentageDatasetTrain)/100.0)
	
	train_x = inputVectors[0:datasetNumExamplesTrain, :]
	test_x = inputVectors[-datasetNumExamplesTest:, :]	
		
	return train_x, train_y, test_x, test_y
	
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code [not used by LUANN, retained for cross compatibility];
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
			
			#print("f = ", f)
	
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			trainDataIndex = 0

			#greedy code [not used by LUANN, retained for cross compatibility];
			for l in range(1, maxLayer+1):
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					for networkIndex in range(1, maxNetwork+1):
						#print("networkIndex = ", networkIndex)
						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						trainBatch(e, batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)
						
					#trainMultipleNetworks code;
					if(l == maxLayer):
						if(trainMultipleNetworks):
							#train combined network final layer
							trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display)

				#trainMultipleNetworks code;
				if(trainMultipleNetworks and (l == maxLayer)):
					testBatchAllNetworksFinalLayer(testBatchX, testBatchY, datasetNumClasses, numberOfLayers)
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

								
def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	#print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
	
if __name__ == "__main__":
	train(greedy=False, trainMultipleNetworks=trainMultipleNetworks)

