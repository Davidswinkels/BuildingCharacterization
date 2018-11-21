#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0, 1, 2, 3]
buildingClasses = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']

fovs = ['F30', 'F60', 'F90', 'F30_60_90']
randomImagePos = [3552, 5219, 19434, 20358, 14528, 9476, 8983, 8851, 6719, 6410]
selectImageID = [6792, 10258, 114858, 4995, 5198, 12550, 83951, 94709, 5131, 14799, 5311, 21782, 68994, 17961, 44492]
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputFileStem = './result/checkimage/CorrectPredictionImage'

# Load file with misclassified images
predictedImages = pd.read_csv(inputFilePath)

predictionData = {}
# Append information from every model to csv file
for posID in selectImageID:
  imagePos = predictedImages.index[predictedImages['ID'] == posID].tolist()
  for buildingClass in buildingClasses:
    measuredValue = int(predictedImages[buildingClass].iloc[imagePos])
    predictionBuildingClassList = [measuredValue]
    print(measuredValue)

    for fov in fovs:
      countCorrectPredictions = 0
      for iteration in iterations:
        predictedColumnName = 'pred_' + buildingClass + '_' + fov + '_' + str(iteration)
        print(predictedColumnName)
        predictedValue = int(predictedImages[predictedColumnName].iloc[imagePos])
        print(predictedValue)
        if int(predictedValue) == int(measuredValue):
          countCorrectPredictions += 1
      predictionBuildingClassList.append(str(countCorrectPredictions) + '/4')
    print(predictionBuildingClassList)
    predictionData[buildingClass] = predictionBuildingClassList
  predictionDF = pd.DataFrame(data=predictionData, columns = buildingClasses)
  buildingID = '_B' + str(predictedImages['BuildingID'].iloc[imagePos].values[0])
  panoID = '_P' + str(predictedImages['pano_id'].iloc[imagePos].values[0])
  neighID = '_N' + str(predictedImages['BU_CODE'].iloc[imagePos].values[0])[-4:]
  print(neighID)
  outputFilePath = outputFileStem + str(posID) + str(buildingID) + str(panoID) + str(neighID) + '.csv'
  predictionDF.to_csv(outputFilePath, index=False)
  print(predictionDF.head())


