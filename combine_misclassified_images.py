#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import csv
import os
import pandas as pd
import numpy as np

print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0, 1, 2, 3]
building_classes = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']
architectures = ['inception_v3']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']

# Create result file name
result_file_path = './result/BuildingPointsPredictedLabels.csv'
building_labels_file_path = './input/BuildingPointsNeighbourhoodCategoriesIteration0.csv'

# Read csv as pandas dataframe
building_labels = pd.read_csv(building_labels_file_path)

# Select testing sample out of population
building_labels_testing = building_labels.loc[building_labels['category'] == 'testing'].copy() #select only buildings used for testing

# Create column that sums all mistakes
building_labels_testing['misclass_sum'] = 0

# Append information from every model to csv file
for building_class in building_classes:
  for fov in fovs:
    for architecture in architectures:
      for iteration in iterations:    
        misclass_file_path = './log/misclass_' + building_class + '_' + fov + '_' + architecture + '_' + str(iteration) + '.csv'
        with open(misclass_file_path, 'r') as stats_file:
          reader = csv.reader(stats_file)
          reader.next() # skip header or first row of file
          predicted_column_name = 'pred_' + building_class + '_' + fov + '_' + str(iteration)
          for row in reader:
            building_labels_testing[predicted_column_name] = building_labels_testing[building_class]
            building_labels_testing.loc[building_labels_testing.BuildingID == int(row[0]), predicted_column_name] = (building_labels_testing[building_class] - 1)**2
          building_labels_testing['misclass_sum'] = np.where((building_labels_testing[building_class] == building_labels_testing[predicted_column_name]), building_labels_testing['misclass_sum'] + 1, building_labels_testing['misclass_sum']) # if ground truth label and predicted label match, then add +1 to misclass_sum in that row, otherwise do not add anything

      
building_labels_testing.to_csv(result_file_path)
print('Result file of misclassified images created at:', result_file_path)
