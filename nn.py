# read the mnist data as an array of images
# write feed forward and backward loop
# input an image. Do feedforward. Do backward loop
# once all this is done, check accuracy on test dataset

import pickle
import numpy as np

n = 784
m = 30 
l = 10
LRate = 0.1
trainSize = 500
testSize = 200

HWeight = np.random.rand(n, m)
HBias = np.random.rand(m)
HActivation = np.zeros(m)

OWeight = np.random.rand(m,l)
OBias = np.random.rand(l)
OActivation = np.zeros(l)

with open('neural-networks-and-deep-learning/data/mnist.pkl') as f:
    data = pickle.load(f)

trainX = data[0][0]
rawTrainY = data[0][1]
validationX = data[1][0]
rawValidationY = data[1][1]
testX = data[2][0]
rawTestY = data[2][1]

trainX = trainX[0:trainSize]
rawTrainY = rawTrainY[0:trainSize]
testX = testX[0:testSize]
rawTestY = rawTestY[0:testSize]

def oneHotRep(inArray):
    outArray = np.zeros((len(inArray), 10))
    for i in range(len(inArray)):
        outArray[i, inArray[i]] = 1
    return outArray

trainY = oneHotRep(rawTrainY)
validationY = oneHotRep(rawValidationY) 
testY = oneHotRep(rawTestY)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def feedForward(x):
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
    print "Before backprop update"
    print "OWeight", OWeight
    print "OBias", OBias
    print "HWeight", HWeight
    print "HBias", HBias

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

c = 1 
for x, y in zip(trainX, trainY):
    print "============== ", c, " ==============="
    print "feedForward ", c
    feedForward(x)
    print "backProp ", c
    backProp(x, y)
    c+=1
    print "============== ", c, " END ===============\n"


# Check test accuracy
def getPrediction(x):
    feedForward(x)
    return OActivation.argmax()

# Test accuracy
def calcAccuracy(X, rawY):
    correct = 0
    predCounts = np.zeros(10)
    for x, y in zip(X, rawY):
        pred = getPrediction(x)
        if  pred == y:
            correct+=1
        predCounts[int(pred)] += 1
    return (correct*1.0)/len(rawTestY), predCounts

trainAcc, trainPredCounts = calcAccuracy(trainX[1:testSize], rawTrainY[1:testSize])
testAcc, testPredCounts = calcAccuracy(testX, rawTestY)
print "Train Accuracy (first testSize elements)========> ", trainAcc, trainPredCounts
print "Test Accuracy ========> ", testAcc, testPredCounts