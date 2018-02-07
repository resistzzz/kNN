from numpy import *
from os import listdir
import operator

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def handwriteClassTest():
    hwLabels = []
    k = 5
    trainFileList = listdir('trainingDigits')
    trainFileListSize = len(trainFileList)
    trainMat = zeros((trainFileListSize, 1024))
    for i in range(trainFileListSize):
        fileNameStr = trainFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector('trainingDigits/' + fileNameStr)
    testFileList = listdir('testDigits')
    testFileListSize = len(testFileList)
    error = 0.0
    for i in range(testFileListSize):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        testVector = img2vector('testDigits/' + fileNameStr)
        classifierResult = classify0(testVector, trainMat, hwLabels, k)
        if classifierResult != classNumStr:
            error += 1
        print('classify result:%d, real result:%d' %(classifierResult, classNumStr))
    errorRate = error/float(testFileListSize)
    print('error number is: %d' %error)
    print('error rate is: %f' %errorRate)

if __name__ == '__main__':
    handwriteClassTest()