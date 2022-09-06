# -*- coding: utf-8 -*-
"""MNISTsparseNetwork.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uzhVwk_dgEcB2Q3CqwZMVBEwJETolwga

# MNIST Sparse Network (SciKit-Learn and skorch)

derived from https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb
"""

! [ ! -z "$COLAB_GPU" ] && pip install torch scikit-learn==0.20.* skorch

from sklearn.metrics.cluster.supervised import fowlkes_mallows_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings("ignore")    #category=DeprecationWarning, message='is a deprecated alias'

useSparseNetwork = True
if(useSparseNetwork):    
    paralleliseSparseProcessing = True   #parallel processing of sparse filters using Conv1d/Conv2d groups parameter
    if(paralleliseSparseProcessing):
        paralleliseSparseProcessingPrintTime = False
        if(paralleliseSparseProcessingPrintTime):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
    
    numberOfEpochsMLP = 1   #default: 10
    numberOfEpochsCNN = 1   #default: 10
    numberOfSparseLayersCNN = 2 #default: 1 (1 or 2)
    numberOfSparseLayersMLP = 2 #default: 1 (1 or 2)
    numberOfSublayerChannelsCNN = 2 #default: 2
    numberOfSublayerChannelsMLP = 2 #default: 2
else:
    numberOfEpochsMLP = 10   #default: 20
    numberOfEpochsCNN = 10   #default: 10
    numberOfSparseLayersCNN = 1 #default: 1 #additional dense hidden layers
    numberOfSparseLayersMLP = 1 #default: 0 #additional dense hidden layers

#set Runtime type = high RAM
#numberOfChannelsFirstDenseLayer: max value determined by numberOfSparseLayers, GPU RAM (independent of batchSize)

#first/dense MLP layer;
if(numberOfSparseLayersMLP == 0):
    numberOfChannelsFirstDenseLayerMLP = 100	#hidden_dim
    batchSizeMLP = 1024 #128
elif(numberOfSparseLayersMLP == 1):
    numberOfChannelsFirstDenseLayerMLP = 100
    batchSizeMLP = 1024 #128
elif(numberOfSparseLayersMLP == 2):
    numberOfChannelsFirstDenseLayerMLP = 20  #20 #2
    batchSizeMLP = 1024 #128
else:
    print("useSparseNetwork warning: numberOfSparseLayersMLP is too high for compute/memory")
    numberOfChannelsFirstDenseLayerMLP = 2
    batchSizeMLP = 16
    
#first/dense CNN layer;
if(numberOfSparseLayersCNN == 0):
    numberOfChannelsFirstDenseLayerCNN = 32
    batchSizeCNN = 4096    #1024
elif(numberOfSparseLayersCNN == 1):
    numberOfChannelsFirstDenseLayerCNN = 32
    batchSizeCNN = 4096    #1024
elif(numberOfSparseLayersCNN == 2):
    numberOfChannelsFirstDenseLayerCNN = 8  #8  #2
    batchSizeCNN = 4096    #1024
else:
    print("useSparseNetwork warning: numberOfSparseLayersCNN is too high for compute/memory")
    numberOfChannelsFirstDenseLayerCNN = 2
    batchSizeCNN = 16

learningAlgorithmLUANN = False
onlyTrainFinalLayer = False #initialise dependent var
if(learningAlgorithmLUANN):
    onlyTrainFinalLayer = True

"""## Sparse Layer Processing

"""

class SparseLayerProcessing():
    def __init__(self, isCNNmodel, numberOfSublayerChannels, numberOfSparseLayers, layerDropout, numberOfChannelsFirstDenseLayer, kernelSize=None, padding=None, stride=None, maxPoolSize=None):

        self.isCNNmodel = isCNNmodel

        self.numberOfSublayerChannels = numberOfSublayerChannels
        self.numberOfSparseLayers = numberOfSparseLayers
        self.layerDropout = layerDropout
        self.numberOfChannelsFirstDenseLayer = numberOfChannelsFirstDenseLayer
        self.sparseLayerList = [None]*self.numberOfSparseLayers

        if(isCNNmodel):
            self.kernelSize = kernelSize
            self.padding = padding
            self.stride = stride
            self.maxPoolSize = maxPoolSize

    def generateSparseLayers(self, numberOfChannels, height=None, width=None):
        for layerIndex in range(self.numberOfSparseLayers):
            #print("layerIndex = ", layerIndex)
            if(useSparseNetwork):
                layer, numberOfChannels = self.generateSparseLayer(numberOfChannels)
                self.sparseLayerList[layerIndex] = layer
                if(self.isCNNmodel):
                    height, width = self.getImageDimensionsAfterConv(height, width, self.kernelSize, self.padding, self.stride, self.maxPoolSize)
            else:
                #only used by CNN originally:
                numberOfInputChannels = numberOfChannels
                numberOfOutputChannels = numberOfChannels*2
                layer = self.generateLayerStandard(numberOfChannels, numberOfOutputChannels)
                self.sparseLayerList[layerIndex] = layer
                numberOfChannels = numberOfOutputChannels
                if(self.isCNNmodel):
                    height, width = self.getImageDimensionsAfterConv(height, width, self.kernelSize, self.padding, self.stride, self.maxPoolSize)
                
        return numberOfChannels, height, width

    def generateSparseLayer(self, numberOfChannels):
        numChannelPairs = self.calculateNumberChannelPairs(numberOfChannels)
        #print("numberOfChannels = ", numberOfChannels)
        #print("numChannelPairs = ", numChannelPairs)
        numberOfInputChannels = self.numberOfSublayerChannels
        numberOfOutputChannels = 1
        if(paralleliseSparseProcessing):
            layer = self.generateSparseLayerParallel(numChannelPairs, numberOfInputChannels, numberOfOutputChannels)
        else:
            layer = self.generateSparseLayerStandard(numChannelPairs, numberOfInputChannels, numberOfOutputChannels)
        numberOfChannels = numChannelPairs*numberOfOutputChannels
        return layer, numberOfChannels
    def generateSparseLayerStandard(self, numChannelPairs, numberOfInputChannels, numberOfOutputChannels):
        sparseSublayerList = []
        for channelPairIndex in range(numChannelPairs):
            sublayer = self.generateLayerStandard(numberOfInputChannels, numberOfOutputChannels)
            sparseSublayerList.append(sublayer)
        return sparseSublayerList
    def generateSparseLayerParallel(self, numChannelPairs, numberOfInputChannels, numberOfOutputChannels):
        layer = self.generateLayerParallel(numChannelPairs, numberOfInputChannels, numberOfOutputChannels)
        return layer

    def generateLayerParallel(self, numChannelPairs, numberOfInputChannels, numberOfOutputChannels):
        if(self.isCNNmodel):
            return self.generateLayerParallelCNN(numChannelPairs, numberOfInputChannels, numberOfOutputChannels)
        else:
            return self.generateLayerParallelMLP(numChannelPairs, numberOfInputChannels, numberOfOutputChannels)
    def generateLayerParallelMLP(self, numChannelPairs, numberOfInputChannels, numberOfOutputChannels):
        #https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch/58389075#58389075
        layer = nn.Conv1d(numberOfInputChannels*numChannelPairs, numberOfOutputChannels*numChannelPairs, kernel_size=1, groups=numChannelPairs)
        return layer
    def generateLayerParallelCNN(self, numChannelPairs, numberOfInputChannels, numberOfOutputChannels):
        conv2DnumberSubChannels = numberOfInputChannels*numChannelPairs
        layer = nn.Conv2d(conv2DnumberSubChannels, conv2DnumberSubChannels, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride, groups=conv2DnumberSubChannels)
        return layer

    def generateLayerStandard(self, numberOfInputChannels, numberOfOutputChannels):
        if(self.isCNNmodel):
            return self.generateLayerStandardCNN(numberOfInputChannels, numberOfOutputChannels)
        else:
            return self.generateLayerStandardMLP(numberOfInputChannels, numberOfOutputChannels)
    def generateLayerStandardMLP(self, numberOfInputChannels, numberOfOutputChannels):
        layer = nn.Linear(numberOfInputChannels, numberOfOutputChannels)
        return layer
    def generateLayerStandardCNN(self, numberOfInputChannels, numberOfOutputChannels):
        layer = nn.Conv2d(numberOfInputChannels, numberOfOutputChannels, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride)
        return layer

    def executeSparseLayers(self, X):
        numberOfChannels = self.numberOfChannelsFirstDenseLayer
        for layerIndex in range(self.numberOfSparseLayers):
            if(useSparseNetwork):
                layerZ, numberOfChannels = self.executeSparseLayer(layerIndex, X, numberOfChannels)
            else:
                layerIn = X
                layerZ = (self.sparseLayerList[layerIndex])(layerIn)
            layerOut = self.activationFunction(layerZ)
            X = layerOut
        return X
    
    def executeSparseLayer(self, layerIndex, X, numberOfChannels):
        numChannelPairs = self.calculateNumberChannelPairs(numberOfChannels)
        numberOfInputChannels = self.numberOfSublayerChannels
        numberOfOutputChannels = 1
        channelsPairsList = []
        self.convertToChannelsToChannelPairsList(X, 0, self.numberOfSublayerChannels, None, channelsPairsList, numberOfChannels)
        if(paralleliseSparseProcessing):
            layerZ = self.executeSparseLayerParallel(layerIndex, channelsPairsList, numChannelPairs)
        else:
            layerZ = self.executeSparseLayerStandard(layerIndex, channelsPairsList, numChannelPairs)
        numberOfChannels = numChannelPairs*numberOfOutputChannels
        return layerZ, numberOfChannels
    def executeSparseLayerStandard(self, layerIndex, channelsPairsList, numChannelPairs):
        channelPairSublayerOutputList = []
        for channelPairIndex in range(numChannelPairs):
            sublayerIn = channelsPairsList[channelPairIndex]
            sublayerOut = (self.sparseLayerList[layerIndex])[channelPairIndex](sublayerIn)
            sublayerOut = torch.squeeze(sublayerOut, dim=1)   #remove channel dim (size=numberOfOutputChannels=1); prepare for convertChannelPairLINoutputListToChannels execution
            channelPairSublayerOutputList.append(sublayerOut)
        layerZ = self.convertChannelPairSublayerOutputListToChannels(channelPairSublayerOutputList)
        return layerZ
    def executeSparseLayerParallel(self, layerIndex, channelsPairsList, numChannelPairs):
        firstTensorInList = channelsPairsList[0]    #shape = [batchSize, numberOfInputChannels, ..]
        print("executeSparseLayerParallel: layerIndex = ", layerIndex, ", firstTensorInList.shape = ", firstTensorInList.shape, ", numChannelPairs = ", numChannelPairs)
        tensorPropertiesTuple = self.getSublayerTensorProperties(firstTensorInList)   #get properties from first tensor in list
        #numChannelPairs = len(channelsPairsList)
        channelsPairs = torch.stack(channelsPairsList, dim=1)   #shape = [batchSize, numChannelPairs, numberOfInputChannels, ..]
        if(self.isCNNmodel):
            if(paralleliseSparseProcessingPrintTime):
                start.record()
            (batchSize, numberOfInputChannels, height, width) = tensorPropertiesTuple
            conv2DnumberSubChannels = numberOfInputChannels*numChannelPairs
            layerIn = torch.reshape(channelsPairs, (batchSize, numChannelPairs*numberOfInputChannels, height, width))
            layerZ = (self.sparseLayerList[layerIndex])(layerIn)  #channels convoluted separately (in separate groups)
            height, width = self.getImageDimensionsAfterConv(height, width, self.kernelSize, self.padding, self.stride, 1)  #no max pool has been performed
            layerZ = torch.reshape(layerZ, (batchSize, numChannelPairs, numberOfInputChannels, height, width))
            layerZ = torch.sum(layerZ, dim=2)  #take sum of numberOfInputChannels (emulates element-wise sum as performed by CNN with groups=1)
            if(paralleliseSparseProcessingPrintTime):
                end.record()
                torch.cuda.synchronize()
                print(start.elapsed_time(end))
        else:
            (batchSize, numberOfInputChannels) = tensorPropertiesTuple
            #https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch/58389075#58389075
            layerIn = torch.reshape(channelsPairs, (batchSize, numChannelPairs*numberOfInputChannels, 1))
            layerZ = (self.sparseLayerList[layerIndex])(layerIn)
            layerZ = torch.reshape(layerZ, (batchSize, numChannelPairs))
        print("executeSparseLayerParallel: layerZ.shape = ", layerZ.shape)
        #layerZ shape = [batchSize, numChannelPairs, ..]
        return layerZ

    def activationFunction(self, Z, useDropOut=True):
        if(self.isCNNmodel):
            return self.activationFunctionCNN(Z, useDropOut)
        else:
            return self.activationFunctionMLP(Z, useDropOut)
    def activationFunctionMLP(self, Z, useDropOut=True):
        A = F.relu(Z)
        if(useDropOut):
            A = self.layerDropout(A)
        return A
    def activationFunctionCNN(self, Z, useDropOut=True):
        if(useDropOut):
            Z = self.layerDropout(Z)
        A = torch.relu(F.max_pool2d(Z, kernel_size=self.maxPoolSize))
        return A

    def calculateNumberChannelPairs(self, numInputChannels):
        numChannelPairs = numInputChannels**self.numberOfSublayerChannels
        return numChannelPairs
        #numOutputChannels = number of filters

    def convertToChannelsToChannelPairsList(self, channels, sublayerChannelIndex, numberOfSublayerChannels, channelPair, channelsPairsList, numberOfChannels):
        if(sublayerChannelIndex == numberOfSublayerChannels):
            channelsPairsList.append(channelPair)
        else:
            #numChannelPairs = self.calculateNumberChannelPairs(numberOfChannels)
            for channelIndex in range(numberOfChannels):
                #channelPairIndex = channelIndex1*numChannelPairs + channelIndex2
                channelPairSub1 = channels[:, channelIndex]  #channels[:, channelIndex1, :]
                channelPairSub1 = torch.unsqueeze(channelPairSub1, dim=1)
                if(sublayerChannelIndex == 0):
                    channelPair2 = channelPairSub1
                else:
                    channelPair2 = torch.clone(channelPair)
                    channelPair2 = torch.cat((channelPair, channelPairSub1), dim=1)
                self.convertToChannelsToChannelPairsList(channels, sublayerChannelIndex+1, numberOfSublayerChannels, channelPair2, channelsPairsList, numberOfChannels)

    def getSublayerTensorProperties(self, channels):
        if(self.isCNNmodel):
            return self.getCNNtensorProperties(channels)
        else:
            return self.getMLPtensorProperties(channels)
    def getMLPtensorProperties(self, channels):
        batchSize = channels.shape[0]
        numberOfChannels = channels.shape[1]
        tensorPropertiesTuple = (batchSize, numberOfChannels)
        return tensorPropertiesTuple
    def getCNNtensorProperties(self, channels):
        batchSize = channels.shape[0]
        numberOfChannels = channels.shape[1]
        height = channels.shape[2]
        width = channels.shape[3]
        tensorPropertiesTuple = (batchSize, numberOfChannels, height, width)
        return tensorPropertiesTuple

    def convertChannelPairSublayerOutputListToChannels(self, channelPairSublayerOutputList):
        layerZ = torch.stack(channelPairSublayerOutputList, dim=1)
        return layerZ

    def getImageDimensionsAfterConv(self, inputHeight, inputWidth, kernelSize, padding, stride, maxPoolSize):
        height = (inputHeight - (kernelSize//2 * 2) + padding) // stride // maxPoolSize    #// = integer floor division
        width = (inputWidth - (kernelSize//2 * 2) + padding) // stride // maxPoolSize
        return height, width

"""## Loading Data
Using SciKit-Learns ```fetch_openml``` to load MNIST data.
"""

mnist = fetch_openml('mnist_784', cache=False)

mnist.data.shape

"""## Preprocessing Data"""

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

"""To avoid big weights that deal with the pixel values from between [0, 255], we scale `X` down. A commonly used range is [0, 1]."""

X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape, y_train.shape

"""## MLP Neural Network"""

import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))

"""A Neural network in PyTorch's framework."""

class MLPModel(nn.Module):
    def __init__(self, input_dim=mnist_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.5):
        super(MLPModel, self).__init__()

        self.isCNNmodel = False

        self.dropout = nn.Dropout(dropout)

        #print("hidden_dim = ", hidden_dim)
        self.numberOfSparseLayers = numberOfSparseLayersMLP   #default: 1 (1 or 2)
        self.numberOfChannelsFirstDenseLayer = numberOfChannelsFirstDenseLayerMLP

        numberOfChannels = self.numberOfChannelsFirstDenseLayer 

        self.linear1 = nn.Linear(input_dim, numberOfChannels)  #first/dense linear layer 

        self.sparseLayerProcessing = SparseLayerProcessing(self.isCNNmodel, numberOfSublayerChannelsMLP, self.numberOfSparseLayers, self.dropout, self.numberOfChannelsFirstDenseLayer)

        numberOfChannels, _, _ = self.sparseLayerProcessing.generateSparseLayers(numberOfChannels)

        self.output = nn.Linear(numberOfChannels, output_dim)

    def forward(self, x, **kwargs):

        #first/dense linear layer
        x = self.linear1(x)
        x = self.sparseLayerProcessing.activationFunction(x)

        x = self.sparseLayerProcessing.executeSparseLayers(x)

        if(onlyTrainFinalLayer):
            x = x.detach()

        x = F.softmax(self.output(x), dim=-1)

        return x

from skorch import NeuralNetClassifier

torch.manual_seed(0)

net = NeuralNetClassifier(
    MLPModel,
    max_epochs=numberOfEpochsMLP,
    lr=0.1,
    device=device,
    batch_size=batchSizeMLP,
)

net.fit(X_train, y_train)

"""## Prediction"""

from sklearn.metrics import accuracy_score

y_pred = net.predict(X_test)

accuracy_score(y_test, y_pred)

"""# Convolutional Network"""

XCnn = X.reshape(-1, 1, 28, 28)

XCnn.shape

XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

XCnn_train.shape, y_train.shape

class CNNModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNNModel, self).__init__()

        self.isCNNmodel = True

        height = 28 #MNIST defined
        width = 28  #MNIST defined
        self.kernelSize = 3
        self.padding = 0
        self.stride = 1
        self.maxPoolSize = 2 #assume max pool at each layer

        self.conv2_drop = nn.Dropout2d(p=dropout)

        self.numberOfSparseLayers = numberOfSparseLayersCNN #default: 1 (1 or 2)
        self.numberOfChannelsFirstDenseLayer = numberOfChannelsFirstDenseLayerCNN

        numberOfChannels = self.numberOfChannelsFirstDenseLayer  
        self.conv1 = nn.Conv2d(1, numberOfChannels, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride)  #first/dense linear layer

        self.sparseLayerProcessing = SparseLayerProcessing(self.isCNNmodel, numberOfSublayerChannelsCNN, self.numberOfSparseLayers, self.conv2_drop, self.numberOfChannelsFirstDenseLayer, kernelSize=self.kernelSize, padding=self.padding, stride=self.stride, maxPoolSize=self.maxPoolSize)
        
        height, width = self.sparseLayerProcessing.getImageDimensionsAfterConv(height, width, self.kernelSize, self.padding, self.stride, self.maxPoolSize)
        numberOfChannels, width, height = self.sparseLayerProcessing.generateSparseLayers(numberOfChannels, height, width)

        firstLinearInputSize = numberOfChannels*width*height

        self.fc1 = nn.Linear(firstLinearInputSize, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.conv1(x)
        x = self.sparseLayerProcessing.activationFunction(x, useDropOut=False)

        x = self.sparseLayerProcessing.executeSparseLayers(x)

        if(onlyTrainFinalLayer):
            x = x.detach()

        # flatten over channel, height and width
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)

        return x

torch.manual_seed(0)

cnn = NeuralNetClassifier(
    CNNModel,
    max_epochs=numberOfEpochsCNN,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=device,
    batch_size=batchSizeCNN,
)

cnn.fit(XCnn_train, y_train)

y_pred_cnn = cnn.predict(XCnn_test)

accuracy_score(y_test, y_pred_cnn)