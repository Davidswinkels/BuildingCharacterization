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
selectImageID = [6792, 10258, 114858, 4995, 5198, 12550, 83951, 94709, 5131, 14799, 5311, 21782, 68994, 17961, 44492]
inputFilePath = './result/BuildingPointsPredictedLabels.csv'
outputFileStem = './result/checkimage/CorrectPredictionImage'


# building_labels_testing.to_csv(result_file_path)
print('Load file of misclassified images from:', inputFilePath)
predictedImages = pd.read_csv(inputFilePath)

print(predictedImages.columns.values)

# predictionData = {}
# ## Append information from every model to csv file
# for imagePos in randomImagePos:
#   for buildingClass in buildingClasses:
#     measuredValue = predictedImages[buildingClass].iloc[imagePos]
#     predictionBuildingClassList = [measuredValue]
#
#     for fov in fovs:
#       countCorrectPredictions = 0
#       for iteration in iterations:
#         predictedColumnName = 'pred_' + buildingClass + '_' + fov + '_' + str(iteration)
#         print(predictedColumnName)
#         predictedValue = predictedImages[predictedColumnName].iloc[imagePos]
#         if predictedValue == measuredValue:
#           countCorrectPredictions += 1
#       predictionBuildingClassList.append(countCorrectPredictions)
#     print(predictionBuildingClassList)
#     predictionData[buildingClass] = predictionBuildingClassList
#   predictionDF = pd.DataFrame(data=predictionData, columns = buildingClasses)
#   buildingID = '_B' + str(predictedImages['BuildingID'].iloc[imagePos])
#   panoID = '_P' + str(predictedImages['BuildingID'].iloc[imagePos])
#   outputFilePath = outputFileStem + str(imagePos) + buildingID + panoID + '.csv'
#   predictionDF.to_csv(outputFilePath)
#   print(predictionDF.head())


predictionData = {}
## Append information from every model to csv file
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
      predictionBuildingClassList.append(countCorrectPredictions)
    print(predictionBuildingClassList)
    predictionData[buildingClass] = predictionBuildingClassList
  predictionDF = pd.DataFrame(data=predictionData, columns = buildingClasses)
  buildingID = '_B' + str(predictedImages['BuildingID'].iloc[imagePos])
  panoID = '_P' + str(predictedImages['BuildingID'].iloc[imagePos])
  outputFilePath = outputFileStem + str(posID) + buildingID + panoID + '.csv'
  predictionDF.to_csv(outputFilePath)
  print(predictionDF.head())


