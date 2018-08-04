#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
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
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputFileStem = './result/checkimage/CorrectPredictionImage'


# building_labels_testing.to_csv(result_file_path)
print('Load file of misclassified images from:', inputFilePath)
predictedImages = pd.read_csv(inputFilePath)

print(predictedImages.columns.values)

predictionData = {}
## Append information from every model to csv file
for imagePos in randomImagePos:
  for buildingClass in buildingClasses:
    measuredValue = predictedImages[buildingClass].iloc[imagePos]
    predictionBuildingClassList = [measuredValue]

    for fov in fovs:
      countCorrectPredictions = 0
      for iteration in iterations:
        predictedColumnName = 'pred_' + buildingClass + '_' + fov + '_' + str(iteration)
        print(predictedColumnName)
        predictedValue = predictedImages[predictedColumnName].iloc[imagePos]
        if predictedValue == measuredValue:
          countCorrectPredictions += 1
      predictionBuildingClassList.append(countCorrectPredictions)
    print(predictionBuildingClassList)
    predictionData[buildingClass] = predictionBuildingClassList
  predictionDF = pd.DataFrame(data=predictionData, columns = buildingClasses)
  buildingID = '_B' + str(predictedImages['BuildingID'].iloc[imagePos])
  panoID = '_P' + str(predictedImages['BuildingID'].iloc[imagePos])
  outputFilePath = outputFileStem + str(imagePos) +  '.csv'
  predictionDF.to_csv(outputFilePath)
  print(predictionDF.head())




