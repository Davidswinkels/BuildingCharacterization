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

# Load predicted labels of building functions
print('Load file of misclassified images from:', inputFilePath)
predictedImages = pd.read_csv(inputFilePath)
print(predictedImages.columns.values)

# Calculate distance, building age and image age to correct formats
predictedImages['distance'] = predictedImages['distance'].str.replace(' m', '')
predictedImages['distance'].apply(ast.literal_eval)
predictedImages['distance'] = pd.to_numeric(predictedImages['distance'])
predictedImages['buildingAge'] = 2016 - predictedImages['Bouwjaar']
predictedImages['imageAge'] = predictedImages['pano_date'].apply(calcDateTime)

# Calculate correct predictions
predictedImages['CorrectPredictions'] = 80 - predictedImages['misclass_sum']
correctPredictionsMean = predictedImages['CorrectPredictions'].mean()
predictedImages['CorrectPredictionsCategory'] = "Correct"
predictedImages.loc[predictedImages.CorrectPredictions<=correctPredictionsMean, 'CorrectPredictionsCategory'] = "Incorrect"
print(predictedImages['CorrectPredictionsCategory'].value_counts())

# Perform T tests with distance, building age and image age as independent variables and correct predictions as dependent variable
t2, p2 = stats.ttest_ind(predictedImages['distance'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['distance'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("Distance: t = " + str(t2))
print("Distance: p = " + str(p2))

t2, p2 = stats.ttest_ind(predictedImages['buildingAge'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['buildingAge'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("Building age: t = " + str(t2))
print("Building age: p = " + str(p2))

t2, p2 = stats.ttest_ind(predictedImages['imageAge'].where(predictedImages["CorrectPredictionsCategory"] == "Correct").dropna(),
                         predictedImages['imageAge'].where(predictedImages["CorrectPredictionsCategory"] == "Incorrect").dropna())
print("Image age: t = " + str(t2))
print("Image age: p = " + str(p2))


# Perform ANOVA with neighbourhoods as independent variable and correct predictions as dependent variable
f, p = stats.f_oneway(predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636501').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638402').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637803').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636101').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634805').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637402').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638900').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634300').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631403').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633705').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630704').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632503').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638901').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636102').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633304').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632802').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639400').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03635503').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634702').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632400').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630303').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634500').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03635300').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632701').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632800').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630501').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03635203').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636802').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631902').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633803').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630706').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639606').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03635605').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639300').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638801').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634801').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630705').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638103').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637301').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630306').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633903').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639605').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634102').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637102').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636912').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631003').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631400').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631304').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632201').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639308').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636900').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633302').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633601').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634405').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630902').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630305').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630903').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630304').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03635502').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03638102').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636100').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639600').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639602').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639305').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639601').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639409').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630703').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631002').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631100').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636902').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637201').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630609').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633600').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636909').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633405').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631007').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631001').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639204').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630701').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03633303').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03634803').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637300').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639500').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636911').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03637202').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03636910').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632300').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03631006').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632302').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639307').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632702').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03632304').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639304').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03630400').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639502').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639206').dropna(),
                      predictedImages['CorrectPredictions'].where(predictedImages['BU_CODE'] == 'BU03639205').dropna())
print("Neighbourhoods: f = " + str(f))
print("Neighbourhoods: p = " + str(p))


