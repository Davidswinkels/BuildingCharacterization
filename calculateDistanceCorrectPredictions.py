#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import pandas as pd
import numpy as np
import ast

def strRound(number1, number2):
  return str(round(number1, 2)) + ' (' + str(round(number2, 2)) + ')'

print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0]
buildingClasses = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']

architectures = ['inception_v3']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputDistFilePath = './result/DistanceCorrectPredictions.csv'
outputAgeFilePath = './result/AgeCorrectPredictions.csv'


# building_labels_testing.to_csv(result_file_path)
print('Load file of misclassified images from:', inputFilePath)
predictedImages = pd.read_csv(inputFilePath)

## Calculate distance from strings to integers
predictedImages['distance'] = predictedImages['distance'].str.replace(' m', '')
predictedImages['distance'].apply(ast.literal_eval)
predictedImages['distance'] = pd.to_numeric(predictedImages['distance'])

print(type(predictedImages['Bouwjaar'].iloc[0]))
print(type(predictedImages['heading'].iloc[0]))
print(predictedImages.columns.values)


outputDistData = {}
outputAgeData = {}

## Append information from every model to csv file
for buildingClass in buildingClasses:
  outputCorrectColumnName = buildingClass[:4] + 'Correct'
  outputIncorrectColumnName = buildingClass[:4] + 'Incorrect'
  DistAvgCorrList = []
  DistAvgIncorrList = []
  AgeAvgCorrList = []
  AgeAvgIncorrList = []
  for fov in fovs:
    for architecture in architectures:
      for iteration in iterations:
        predictedColumnName = 'pred_' + buildingClass + '_' + fov + '_' + str(iteration)
        print(predictedColumnName)
        corrDistAvg = predictedImages['distance'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().mean()
        corrDistStd = predictedImages['distance'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().std()
        incorrDistAvg = predictedImages['distance'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().mean()
        incorrDistStd = predictedImages['distance'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().std()
        corrAgeAvg = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().mean()
        corrAgeStd = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().std()
        incorrAgeAvg = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().mean()
        incorrAgeStd = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().std()
        DistAvgCorrList.append(strRound(corrDistAvg, corrDistStd))
        DistAvgIncorrList.append(strRound(incorrDistAvg, incorrDistStd))
        AgeAvgCorrList.append(strRound(corrAgeAvg, corrAgeStd))
        AgeAvgIncorrList.append(strRound(incorrAgeAvg, incorrAgeStd))
  outputDistData[outputCorrectColumnName] = DistAvgCorrList
  outputDistData[outputIncorrectColumnName] = DistAvgIncorrList
  outputAgeData[outputCorrectColumnName] = AgeAvgCorrList
  outputAgeData[outputIncorrectColumnName] = AgeAvgIncorrList

print(outputDistData)
print(outputAgeData)

distDF = pd.DataFrame(data=outputDistData)
ageDF = pd.DataFrame(data=outputAgeData)

distDF.to_csv(outputDistFilePath)
ageDF.to_csv(outputAgeFilePath)



