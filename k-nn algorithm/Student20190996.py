#!/usr/bin/python3

import sys
import os

import numpy as np
import operator

def createDataSet(trainingDataFolder, testDataFolder):
	trainingSet = []
	testSet = []
	trainingLabels = []
	testLabels = []

	path = os.getcwd() + '\\' + trainingDataFolder
	for file in os.listdir(path):
		text = ''
		f = open(trainingDataFolder + '\\' + file, "rt")

		# 줄 단위로 읽어오기, \n제거
		while (True):
			line = f.readline()
			if len(line) == 0:
				break
			line = line.strip()
			text += line
		f.close()
		tList = list(text)
		# '0'-> 0, '1'-> 1
		for i in range(len(tList)):
			tList[i] = int(tList[i])

		# trainingSet에 요소 추가, label 정답 추가
		trainingSet.append(tList)
		trainingLabels.append(int(file[0]))

	path = os.getcwd() + '\\' + testDataFolder
	for file in os.listdir(path) :
		text = ''
		f = open(testDataFolder + '\\' + file, "rt")

		while (True):
			line = f.readline()
			if len(line) == 0:
				break
			line = line.strip()
			text += line
		f.close()
		tList = list(text)
		for i in range(len(tList)):
			tList[i] = int(tList[i])

		testSet.append(tList)
		testLabels.append(int(file[0]))

	return trainingSet, testSet, trainingLabels, testLabels


def classify0(testObj, trainingSet, labels, k):
	dataSet = np.array(trainingSet)
	inX = np.array(testObj)

	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	distances = (diffMat ** 2).sum(axis = 1) ** 0.5
	sortedDistIndicies = distances.argsort()

	classCount = {}
	voteIlabel = -1
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		if voteIlabel not in classCount.keys():
			classCount[voteIlabel] = 0
		classCount[voteIlabel] += 1

	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


# main
trainingDataFolder = sys.argv[1]
testDataFolder = sys.argv[2]
trainingSet, testSet, trainingLabels, testLabels = createDataSet(trainingDataFolder, testDataFolder)

for k in range(20):
	fail = 0
	for i in range(len(testLabels)):
		calcX = classify0(testSet[i], trainingSet, trainingLabels, k + 1)
		if testLabels[i] != calcX:
			fail += 1
	print(int(float(fail) / float(len(testLabels)) * 100))
    #print("k = {}, {}".format(k + 1, fail / len(testLabels) * 100))