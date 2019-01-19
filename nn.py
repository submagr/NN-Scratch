# read the mnist data as an array of images
# write feed forward and backward loop
# input an image. Do feedforward. Do backward loop
# once all this is done, check accuracy on test dataset

import pickle
import numpy as np

n = 784
m = 20
l = 10
LRate = 1 
# trainSize = 000
# testSize = 100

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

# trainX = trainX[0:trainSize]
# rawTrainY = rawTrainY[0:trainSize]
# testX = testX[0:testSize]
# rawTestY = rawTestY[0:testSize]

def oneHotRep(inArray):
    outArray = np.zeros((len(inArray), 10))
    for i in range(len(inArray)):
        outArray[i, inArray[i]] = 1
    return outArray

trainY = oneHotRep(rawTrainY)
validationY = oneHotRep(rawValidationY) 
testY = oneHotRep(rawTestY)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def feedForward(x):
    for j in range(m):
        zH = HBias[j]
        for i in range(n):
            zH += HWeight[i, j] * x[i]
        HActivation[j] = sigmoid(zH)
    
    for k in range(l):
        zO = OBias[k]
        for j in range(m):
            zO += OWeight[j, k] * HActivation[j]
        OActivation[k] = sigmoid(zO)

def backProp(x, y):
    # Output Weights and Biases
    for k in range(l):
        OBias[k] += LRate*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])
        for j in range(m):
            OWeight[j, k] += LRate*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])*HActivation[j]

    # Hidden Weights and Biases
    for j in range(m):
        temp = 0
        for k in range(l):
            temp += (-1)*(y[k] - OActivation[k])*(OActivation[k])*(1 - OActivation[k])*(OWeight[j, k])
        HBias[j] = temp*HActivation[j]*(1 - HActivation[j])

        for i in range (n):
            HWeight[i, j] = temp*HActivation[j]*(1 - HActivation[j])*x[i]

c = 1 
for x, y in zip(trainX, trainY):
    print "feedForward ", c
    feedForward(x)
    print "backProp ", c
    backProp(x, y)
    c+=1
    print ""


# Check test accuracy
def getPrediction(x):
    feedForward(x)
    return OActivation.argmax()

# Test accuracy
correct = 0
for x, y in zip(testX, rawTestY):
    pred = getPrediction(x)
    if  pred == y:
        correct+=1
        print "Correct"
    else:
        print "Incorrect Output ", pred, " != ", y

print (correct*1.0)/len(rawTestY)