#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

print('Workspace:', os.getcwd())

# Creating input variables for names
iterations = [0, 1, 2, 3]
building_classes = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']
architectures = ['inception_v3','mobilenet_1.0_224']
fovs = ['F30', 'F60', 'F90', 'F30_60_90']

# Create result file name
result_file_name = './result/outcome_statistics_meeting_office.csv'

print('Result file of outcome stats created at:', result_file_name)

# Create csv file with header
with open(result_file_name, 'wb') as result_file:
  wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
  wr.writerow(['building_class', 'fov', 'architecture', 'iteration', 'train_n', 'validation_n', 'test_n','kappa','precision','recall','computation_time(seconds)','test_accuracy', 'average_accuracy', 'f_score'])

# Append information from every model to csv file
for building_class in building_classes:
  for fov in fovs:
    for architecture in architectures:
      for iteration in iterations:
        print('Opening CNN Model: ', building_class, fov, architecture, str(iteration))

        # Open outcome statistics file
        stats_file_name = './log/stats_' + building_class + '_' + fov + '_' + architecture + '_' + str(iteration) + '.csv'
        with open(stats_file_name, 'r') as stats_file:
          reader = csv.reader(stats_file)
          reader.next() # skip header or first row of file
          outcome_stats_data = reader.next()

        # Open confusion matrix file for meeting and office
        if building_class == "Meeting" or building_class == "Office":
          conf_file_name = './log/conf_' + building_class + '_' + fov + '_' + architecture + '_' + str(iteration) + '.csv'
          with open(conf_file_name, 'r') as conf_file:
            reader = csv.reader(conf_file)
            reader.next() # skip header or first row of file
            conf_data_row1 = reader.next()
            conf_data_row2 = reader.next()
            conf_data_row3 = reader.next()
          print(conf_data_row1)
          print(conf_data_row2)
          print(conf_data_row3)

          # Calculate kappa, precision and recall
          proport_correct = (float(conf_data_row2[2]) + float(conf_data_row1[1])) / float(conf_data_row3[3])
          prob_class = ((float(conf_data_row2[2]) + float(conf_data_row2[1])) / float(conf_data_row3[3])) * ((float(conf_data_row2[2]) + float(conf_data_row1[2])) / float(conf_data_row3[3]))
          prob_non_class = ((float(conf_data_row1[2]) + float(conf_data_row1[1])) / float(conf_data_row3[3])) * ((float(conf_data_row2[1]) + float(conf_data_row1[1])) / float(conf_data_row3[3]))
          prob_all = prob_class + prob_non_class
          kappa = (proport_correct - prob_all) / (1 - prob_all)
          precision = float(conf_data_row2[2]) / float(conf_data_row3[2])
          recall = float(conf_data_row2[2]) / float(conf_data_row2[3])

        # Open confusion matrix file for residential, industry and shop
        if building_class == "Residentia" or building_class == "Industry" or building_class == "Shop":
          conf_file_name = './log/conf_' + building_class + '_' + fov + '_' + architecture + '_' + str(iteration) + '.csv'
          with open(conf_file_name, 'r') as conf_file:
            reader = csv.reader(conf_file)
            reader.next() # skip header or first row of file
            conf_data_row1 = reader.next()
            conf_data_row2 = reader.next()
            conf_data_row3 = reader.next()
          print(conf_data_row1)
          print(conf_data_row2)
          print(conf_data_row3)

          # Calculate kappa, precision and recall
          proport_correct = (float(conf_data_row1[1]) + float(conf_data_row2[2])) / float(conf_data_row3[3])
          prob_class = (float(conf_data_row1[3]) / float(conf_data_row3[3])) * (float(conf_data_row3[1]) / float(conf_data_row3[3]))
          prob_non_class = (float(conf_data_row2[3]) / float(conf_data_row3[3])) * (float(conf_data_row3[2]) / float(conf_data_row3[3]))
          prob_all = prob_class + prob_non_class
          kappa = (proport_correct - prob_all) / (1 - prob_all)
          precision = float(conf_data_row1[1]) / float(conf_data_row1[3])
          recall = float(conf_data_row1[1]) / float(conf_data_row3[1])

        # Calc f_score
        f_score = (2 * precision * recall) / (precision + recall)

        # Write data to outcome statistics file
        with open(result_file_name, 'ab') as result_file:
          wr = csv.writer(result_file)
          row = [building_class, fov, architecture, iteration, outcome_stats_data[0], outcome_stats_data[1], outcome_stats_data[2], kappa, precision, recall, outcome_stats_data[6], outcome_stats_data[7], outcome_stats_data[8], f_score]
          wr.writerow(row)
