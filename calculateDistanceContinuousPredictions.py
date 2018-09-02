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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

## Calculate distance from strings to integers
predictedImages['distance'] = predictedImages['distance'].str.replace(' m', '')
predictedImages['distance'].apply(ast.literal_eval)
predictedImages['distance'] = pd.to_numeric(predictedImages['distance'])
predictedImages['Distance to building (meters)'] = predictedImages['distance']

predictedImages['Building age (years)'] = 2016 - predictedImages['Bouwjaar']

predictedImages['Streetview image age (days)'] = predictedImages['pano_date'].apply(calcDateTime)
predictedImages['Correct predictions'] = 80 - predictedImages['misclass_sum']


sns.set(color_codes=True)
x_vars = ["Distance to building (meters)", "Building age (years)", "Streetview image age (days)"]
y_var = ["Correct predictions"]
sns.pairplot(predictedImages, x_vars=x_vars, y_vars=y_var,height=5, aspect=.8, kind="reg");
plt.savefig('./result/images/understandingcharacteristics.png')
plt.show()
