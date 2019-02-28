# read the mnist data as an array of images
# write feed forward and backward loop
# input an image. Do feedforward. Do backward loop
# once all this is done, check accuracy on test dataset

import pickle
import numpy as np
from random import shuffle
from random import uniform

n = 2 
m = 3 
l = 2 
LRate = 0.65
trainSize = 10000
testSize =  int(trainSize*0.2)

np.random.seed(123)
np.set_printoptions(precision=2)

HWeight = np.random.rand(n, m)
HBias = np.random.rand(m)
HActivation = np.zeros(m)

OWeight = np.random.rand(m,l)
OBias = np.random.rand(l)
OActivation = np.zeros(l)


def genLinearData(size):
    data = []
    for i in range(size):
        x = uniform(-1, 1)
        y = 2*x + 3 
        # y = (x+4)*(x-2)
        delta = uniform(-1, 1)
        data.append([x, y + delta, 1 if delta >= 0 else 0]) 
    shuffle(data)
    return data

data = genLinearData(trainSize + testSize)

trainX = [[x[0], x[1]] for x in data[0:trainSize]]
rawTrainY = [x[2] for x in data[0:trainSize]]

testX = [[x[0], x[1]] for x in data[trainSize:trainSize+testSize]]
rawTestY = [x[2] for x in data[trainSize:trainSize+testSize]]

def oneHotRep(inArray, numCategories):
    outArray = np.zeros((len(inArray), numCategories))
    for i in range(len(inArray)):
        outArray[i, inArray[i]] = 1
    return outArray

trainY = oneHotRep(rawTrainY, l)
testY = oneHotRep(rawTestY, l)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def feedForward(x):
    print ">>>>>>>>>> feedForwarding x:"
    print x

    for j in range(m):
        zH = 0
        for i in range(n):
            zH += HWeight[i, j] * x[i]
        zH = zH + HBias[j]
        print j, zH
        HActivation[j] = sigmoid(zH)
    
    for k in range(l):
        zO = 0
        for j in range(m):
            zO += OWeight[j, k] * HActivation[j]
        zO = zO + OBias[k]
        OActivation[k] = sigmoid(zO)
    print 'HActivation: ', HActivation
    print 'OActivation: ', OActivation

def backProp(x, y):
    print ">>>>>>>>>> backProp"
    print y

    # Hidden Weights and Biases
    for j in range(m):
        temp = 0
        for k in range(l):
            temp += LRate*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])*(OWeight[j, k])
        temp *= HActivation[j]*(1 - HActivation[j])
        HBias[j] += temp

        for i in range (n):
            HWeight[i, j] += temp*x[i]

    # Output Weights and Biases
    for k in range(l):
        OBias[k] += LRate*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])
        for j in range(m):
            OWeight[j, k] += LRate*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])*HActivation[j]

    print "After backprop update ====>"
    print "OWeight", OWeight
    print "OBias", OBias
    print "HWeight", HWeight
    print "HBias", HBias

def showState():
    print "\t HWeight"
    print HWeight
    print "\t HBias"
    print HBias

    print "\t OWeight"
    print OWeight
    print "\t OBias"
    print OBias

# c = 1
# for x, y in zip(trainX, trainY):
#     print "============== Sample #", c, "==============="
#     print ">>>>>>>>>> Init State"
#     showState()
#     feedForward(x)
#     backProp(x, y)
#     print ">>>>>>>>>> Final State"
#     showState()
#     print "============== Sample #", c, " ENDS ===============\n"
#     c+=1

batch_size = 10
current_batch = 1
for x, y in zip(trainX, trainY):
    batch_HActivations = []
    batch_OActivations = []
    batch_OWeights = []
    batch_x = []
    batch_y = []

    feedForward(x)
    batch_HActivations.append(HActivation)
    batch_OActivations.append(OActivation)
    batch_OWeights.append(OWeight)
    batch_x.append(x)
    batch_y.append(y)

    if current_batch%batch_size == 0:
        # update weights now

        current_batch_size = len(batch_HActivations)

        # Hidden Weights and Biases
        for i in range(n):
            for j in range(m):
                dWeight_i_j = 0
                for iNum in range(current_batch_size):
                    temp = 0;
                    for k in range(l):
                        temp += (batch_y[iNum][k] - batch_OActivations[iNum][k]) * batch_OActivations[iNum][k] * (1 -
                            batch_OActivations[iNum][k]) * batch_OWeights[iNum][j][k]
                    temp *= batch_HActivations[iNum][j] * (1 - batch_HActivations[iNum][j]) * batch_x[iNum][i]
                    dWeight_i_j += temp
                HWeight[i][j] += (LRate/current_batch_size)*dWeight_i_j

        for j in range(m):
            dbias_j = 0
            for iNum in range(current_batch_size):
                temp = 0;
                for k in range(l):
                    temp += (batch_y[iNum][k] - batch_OActivations[iNum][k]) * batch_OActivations[iNum][k] * (1 -
                        batch_OActivations[iNum][k]) * batch_OWeights[iNum][j][k]
                temp *= batch_HActivations[iNum][j] * (1 - batch_HActivations[iNum][j])
                dbias_j += temp
            HBias[j] += (LRate/current_batch_size)*dbias_j

        # Output Weights and Biases
        for k in range(l):
            temp = 0
            for iNum in range(current_batch_size):
                temp += (batch_y[iNum][k] - batch_OActivations[iNum][k]) * (batch_OActivations[iNum][k]) * (1 -
                         batch_OActivations[iNum][k])
            OBias[k] += (LRate/current_batch_size)*temp

            for j in range(m):
                temp = 0
                for iNum in range(current_batch_size):
                    temp += LRate * (batch_y[iNum][k] - batch_OActivations[iNum][k]) * (batch_OActivations[iNum][k]) *\
                            (1 - batch_OActivations[iNum][k]) * batch_HActivations[iNum][j]
                OWeight[j, k] = (LRate/current_batch_size) * temp

        print "After backprop update ====>"
        print "OWeight", OWeight
        print "OBias", OBias
        print "HWeight", HWeight
        print "HBias", HBias

    current_batch += 1

# Check test accuracy
def getPrediction(x):
    feedForward(x)
    return OActivation.argmax()

# Test accuracy
def calcAccuracy(X, Y, numCategories):
    correct = 0
    predCounts = np.zeros(numCategories)
    for x, y in zip(X, Y):
        pred = getPrediction(x)
        if  pred == y:
            correct+=1
        predCounts[int(pred)] += 1
    return (correct*1.0)/len(Y), predCounts

trainAcc, trainPredCounts = calcAccuracy(trainX[1:testSize], rawTrainY[1:testSize], l)
testAcc, testPredCounts = calcAccuracy(testX, rawTestY, l)
print "Train Accuracy (first testSize elements)========> ", trainAcc, trainPredCounts
print "Test Accuracy ========> ", testAcc, testPredCounts