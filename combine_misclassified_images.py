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

print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0, 1, 2, 3]
building_classes = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']
architectures = ['inception_v3']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']

# Create result file name
result_file_path = './result/misclassified_images_inception.csv'
building_labels_file_path = './input/BuildingPointsValidImages.csv'

# Read csv as pandas dataframe
building_labels = pd.read_csv(building_labels_file_path)

# Select testing sample out of population
building_labels_testing = building_labels.loc[building_labels['valid'] == 'No'] #select only buildings used for testing
print('Result file of misclassified images created at:', result_file_path)

# Create csv file with header
with open(result_file_path, 'wb') as result_file:
  wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
  wr.writerow(['building_class', 'fov', 'architecture', 'iteration', 'train_n', 'validation_n', 'test_n','kappa','precision','recall','computation_time(seconds)','test_accuracy', 'average_accuracy', 'f_score'])

# Append information from every model to csv file
for building_class in building_classes:
  for fov in fovs:
    for architecture in architectures:
      for iteration in iterations:    
        misclass_file_path = './log/misclass_' + building_class + '_' + fov + '_' + architecture + '_' + str(iteration) + '.csv'
        with open(misclass_file_path, 'r') as stats_file:
          reader = csv.reader(stats_file)
          reader.next() # skip header or first row of file
          misclass_column_name = 'misclass_' + building_class + '_' + fov + '_' + str(iteration)
          for row in reader:
            building_labels_testing[misclass_column_name] = building_labels_testing[building_class]
            building_labels_testing.loc[building_labels_testing.BuildingID == int(row[0]), misclass_column_name] = (building_labels_testing[building_class] - 1)**2
            print("New row .... ")
            print(" ".join(row))
          # Calc f_score as (2 * precision * recall) / (precision + recall)
          #f_score = (2 * float(outcome_stats_data[4]) * float(outcome_stats_data[5])) / (float(outcome_stats_data[4]) + float(outcome_stats_data[5]))
          #with open(result_file_name, 'ab') as result_file:
          #  wr = csv.writer(result_file)
          #  row = [building_class, fov, architecture, iteration, outcome_stats_data[0], outcome_stats_data[1], outcome_stats_data[2], outcome_stats_data[3], outcome_stats_data[4], outcome_stats_data[5], outcome_stats_data[6], outcome_stats_data[7], outcome_stats_data[8], f_score] 
          #  wr.writerow(row)
          
building_labels_testing.to_csv(result_file_path)
