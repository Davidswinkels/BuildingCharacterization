#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import pandas as pd
import ast
from dateutil import relativedelta as rdelta
from datetime import date
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def strRound(number1, number2):
  return str(round(number1, 2)) + ' (' + str(round(number2, 2)) + ')'

def calcDateTime(dateText):
  year = dateText[:4]
  month = dateText[5:7]
  dateDifference = date(2016,11,1) - date(int(year), int(month), 1)
  return(dateDifference.days)


print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0]
buildingClasses = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']

architectures = ['inception_v3']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputDistFilePath = './result/DistanceCorrectPredictions.csv'
outputBuildAgeFilePath = './result/BuildAgeCorrectPredictions.csv'
outputImageAgeFilePath = './result/ImageAgeCorrectPredictions.csv'

# building_labels_testing.to_csv(result_file_path)
print('Load file of misclassified images from:', inputFilePath)
predictedImages = pd.read_csv(inputFilePath)

print(predictedImages.columns.values)

predictedImages['distance'] = predictedImages['distance'].str.replace(' m', '')
predictedImages['distance'].apply(ast.literal_eval)
predictedImages['distance'] = pd.to_numeric(predictedImages['distance'])
predictedImages['buildingAge'] = 2016 - predictedImages['Bouwjaar']
predictedImages['imageAge'] = predictedImages['pano_date'].apply(calcDateTime)

# x = predictedImages[['distance', 'buildingAge', 'imageAge']]
# print(x.columns.values)
#
# lm = LinearRegression()
# lm.fit(x, predictedImages['CorrectPredictions'])
#
# print(lm.intercept_)
# print(len(lm.coef_))
#
# regressionOutput = pd.DataFrame(zip(x.columns, lm.coef_), columns = ['features', 'estimatedCoefficients'])
# print(regressionOutput.head())
#
# plt.scatter(predictedImages.distance, predictedImages.CorrectPredictions)
# plt.title("Distance versus correct predictions")
# plt.show()
# plt.scatter(predictedImages.buildingAge, predictedImages.CorrectPredictions)
# plt.title("Building age versus correct predictions")
# plt.show()
# plt.scatter(predictedImages.imageAge, predictedImages.CorrectPredictions)
# plt.title("Image age versus correct predictions")
# plt.show()
# plt.scatter(predictedImages.imageAge, predictedImages.distance)
# plt.title("Image age versus correct distance")
# plt.show()
# plt.scatter(predictedImages.imageAge, predictedImages.buildingAge)
# plt.title("Image age versus correct building age")
# plt.show()

predictedImages['CorrectPredictions'] = 80 - predictedImages['misclass_sum']
correctPredictionsMean = predictedImages['CorrectPredictions'].mean()
predictedImages['CorrectPredictionsCategory'] = "Correct"
predictedImages.loc[predictedImages.CorrectPredictions<=correctPredictionsMean, 'CorrectPredictionsCategory'] = "Incorrect"
print(predictedImages['CorrectPredictionsCategory'].value_counts())


t2, p2 = stats.ttest_ind(predictedImages['distance'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['distance'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("t = " + str(t2))
print("p = " + str(p2))

t2, p2 = stats.ttest_ind(predictedImages['buildingAge'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['buildingAge'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("t = " + str(t2))
print("p = " + str(p2))

t2, p2 = stats.ttest_ind(predictedImages['imageAge'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['imageAge'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("t = " + str(t2))
print("p = " + str(p2))



