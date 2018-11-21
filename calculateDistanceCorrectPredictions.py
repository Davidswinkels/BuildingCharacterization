#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import pandas as pd
import ast
from dateutil import relativedelta as rdelta
from datetime import date


def round_numbers_to_string(number1, number2):
  return str(round(number1, 2)) + ' (' + str(round(number2, 2)) + ')'


def calculate_date_time(datetext):
  year = datetext[:4]
  month = datetext[5:7]
  dateDifference = date(2016, 11, 1) - date(int(year), int(month), 1)
  return dateDifference.days



# Creating input variables for names
iterations = [0]
buildingClasses = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']
architectures = ['inception_v3']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputDistFilePath = './result/DistanceCorrectPredictions.csv'
outputBuildAgeFilePath = './result/BuildAgeCorrectPredictions.csv'
outputImageAgeFilePath = './result/ImageAgeCorrectPredictions.csv'

predictedImages = pd.read_csv(inputFilePath)
print('Loaded file with list of misclassified images from:', inputFilePath)
print(predictedImages.columns.values)

# Calculate distance from strings to integers
predictedImages['distance'] = predictedImages['distance'].str.replace(' m', '')
predictedImages['distance'].apply(ast.literal_eval)
predictedImages['distance'] = pd.to_numeric(predictedImages['distance'])
predictedImages['Bouwjaar'] = 2016 - predictedImages['Bouwjaar']
predictedImages['pano_date'] = predictedImages['pano_date'].apply(calculate_date_time)

# Create empty dictionaries to be filled with output data
outputDistData = {}
outputBuildAgeData = {}
outputImageAgeData = {}

# Append information from every model to csv file
for buildingClass in buildingClasses:
  outputCorrectColumnName = buildingClass[:4] + 'Correct'
  outputIncorrectColumnName = buildingClass[:4] + 'Incorrect'
  DistAvgCorrList = []
  DistAvgIncorrList = []
  BuildingAgeAvgCorrList = []
  BuildingAgeAvgIncorrList = []
  ImageAgeAvgCorrList = []
  ImageAgeAvgIncorrList = []
  for fov in fovs:
    for architecture in architectures:
      for iteration in iterations:
        predictedColumnName = 'pred_' + buildingClass + '_' + fov + '_' + str(iteration)
        print(predictedColumnName)
        corrDistAvg = predictedImages['distance'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().mean()
        corrDistStd = predictedImages['distance'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().std()
        incorrDistAvg = predictedImages['distance'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().mean()
        incorrDistStd = predictedImages['distance'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().std()
        corrBuildAgeAvg = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().mean()
        corrBuildAgeStd = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().std()
        incorrBuildAgeAvg = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().mean()
        incorrBuildAgeStd = predictedImages['Bouwjaar'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().std()
        corrImageAgeAvg = predictedImages['pano_date'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().mean()
        corrImageAgeStd = predictedImages['pano_date'].where(predictedImages[buildingClass] == predictedImages[predictedColumnName]).dropna().std()
        incorrImageAgeAvg = predictedImages['pano_date'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().mean()
        incorrImageAgeStd = predictedImages['pano_date'].where(predictedImages[buildingClass] != predictedImages[predictedColumnName]).dropna().std()
        DistAvgCorrList.append(round_numbers_to_string(corrDistAvg, corrDistStd))
        DistAvgIncorrList.append(round_numbers_to_string(incorrDistAvg, incorrDistStd))
        BuildingAgeAvgCorrList.append(round_numbers_to_string(corrBuildAgeAvg, corrBuildAgeStd))
        BuildingAgeAvgIncorrList.append(round_numbers_to_string(incorrBuildAgeAvg, incorrBuildAgeStd))
        ImageAgeAvgCorrList.append(round_numbers_to_string(corrImageAgeAvg, corrImageAgeStd))
        ImageAgeAvgIncorrList.append(round_numbers_to_string(incorrImageAgeAvg, incorrImageAgeStd))
  outputDistData[outputCorrectColumnName] = DistAvgCorrList
  outputDistData[outputIncorrectColumnName] = DistAvgIncorrList
  outputBuildAgeData[outputCorrectColumnName] = BuildingAgeAvgCorrList
  outputBuildAgeData[outputIncorrectColumnName] = BuildingAgeAvgIncorrList
  outputImageAgeData[outputCorrectColumnName] = ImageAgeAvgCorrList
  outputImageAgeData[outputIncorrectColumnName] = ImageAgeAvgIncorrList

# Store dictionary in dataframe and save to file
distDF = pd.DataFrame(data=outputDistData)
ageBuildingDF = pd.DataFrame(data=outputBuildAgeData)
ageImageDF = pd.DataFrame(data=outputImageAgeData)

distDF.to_csv(outputDistFilePath)
ageBuildingDF.to_csv(outputBuildAgeFilePath)
ageImageDF.to_csv(outputImageAgeFilePath)

