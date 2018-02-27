#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import pandas as pd


# Set parameters
input_file = "./input/BuildingPointsValidImages.csv"
building_class = 'Residentia'
fov = 'F30'
iteration = 0

# Load data
building_points = pd.read_csv(input_file)

# Load valid data of specific building class
valid_rows = (building_points['valid'] == 'Yes')
valid_columns = ['ID', 'BuildingID', 'BU_CODE', 'pano_id', building_class]
images_valid = building_points.loc[valid_rows,valid_columns]

# Count amount of buildings per neighbourhood
total_buildings_neighbourhood = images_valid.groupby('BU_CODE')['BU_CODE'].count()
class_buildings_neighbourhood = images_valid.groupby('BU_CODE')[building_class].sum()
non_class_buildings_neighbourhood = total_buildings_neighbourhood - class_buildings_neighbourhood
class_distribution = (float(sum(class_buildings_neighbourhood)) / float(sum(total_buildings_neighbourhood))) * 100.0
print("Class distribution (", building_class, "):", round(class_distribution, 2), "percentage")
# Get unique neighbourhood codes from valid images
neighbourhood_codes = images_valid['BU_CODE'].unique()

# Create five fold list of neighbourhood codes
kfold00_20_list = []
kfold20_40_list = []
kfold40_60_list = []
kfold60_80_list = []
kfold80_100_list = []

# Create five fold list that counts amount of buildings
# Counting building is done for type of class, non-class and total
kfold00_20_count = [0, 0, 0]  # count buildings[class, non-class, total]
kfold20_40_count = [0, 0, 0]  # count buildings[class, non-class, total]
kfold40_60_count = [0, 0, 0]  # count buildings[class, non-class, total]
kfold60_80_count = [0, 0, 0]  # count buildings[class, non-class, total]
kfold80_100_count = [0, 0, 0]  # count buildings[class, non-class, total]

# Create five subsets that have equal distribution with different neighbourhoods
for idx, neighbourhood_code in enumerate(neighbourhood_codes):
  print("Iteration number: " + str(idx + 1))
  # Instantiate five fold subsets with five neighbourhoods each
  if len(kfold00_20_list) < 5:
      kfold00_20_list.append(neighbourhood_code)
      kfold00_20_count[0] = kfold00_20_count[0] + class_buildings_neighbourhood[idx]
      kfold00_20_count[1] = kfold00_20_count[1] + non_class_buildings_neighbourhood[idx]
      kfold00_20_count[2] = kfold00_20_count[2] + total_buildings_neighbourhood[idx]
  elif len(kfold20_40_list) < 5:
      kfold20_40_list.append(neighbourhood_code)
      kfold20_40_count[0] = kfold20_40_count[0] + class_buildings_neighbourhood[idx]
      kfold20_40_count[1] = kfold20_40_count[1] + non_class_buildings_neighbourhood[idx]
      kfold20_40_count[2] = kfold20_40_count[2] + total_buildings_neighbourhood[idx]
  elif len(kfold40_60_list) < 5:
      kfold40_60_list.append(neighbourhood_code)
      kfold40_60_count[0] = kfold40_60_count[0] + class_buildings_neighbourhood[idx]
      kfold40_60_count[1] = kfold40_60_count[1] + non_class_buildings_neighbourhood[idx]
      kfold40_60_count[2] = kfold40_60_count[2] + total_buildings_neighbourhood[idx]
  elif len(kfold60_80_list) < 5:
      kfold60_80_list.append(neighbourhood_code)
      kfold60_80_count[0] = kfold60_80_count[0] + class_buildings_neighbourhood[idx]
      kfold60_80_count[1] = kfold60_80_count[1] + non_class_buildings_neighbourhood[idx]
      kfold60_80_count[2] = kfold60_80_count[2] + total_buildings_neighbourhood[idx]
  elif len(kfold80_100_list) < 5:
      kfold80_100_list.append(neighbourhood_code)
      kfold80_100_count[0] = kfold80_100_count[0] + class_buildings_neighbourhood[idx]
      kfold80_100_count[1] = kfold80_100_count[1] + non_class_buildings_neighbourhood[idx]
      kfold80_100_count[2] = kfold80_100_count[2] + total_buildings_neighbourhood[idx]

  building_total = kfold00_20_count[2] + kfold20_40_count[2] + kfold40_60_count[2] \
                   + kfold60_80_count[2] + kfold80_100_count[2]
  # After instantation fill each of folds to have same number of buildings
  if idx >= 25:
      # Fill five folds to each have 20% of buildings
      if ((float(kfold00_20_count[2]) / float(building_total)) * 100.0) < 20.0:
          kfold00_20_list.append(neighbourhood_code)
          kfold00_20_count[0] = kfold00_20_count[0] + class_buildings_neighbourhood[idx]
          kfold00_20_count[1] = kfold00_20_count[1] + non_class_buildings_neighbourhood[idx]
          kfold00_20_count[2] = kfold00_20_count[2] + total_buildings_neighbourhood[idx]
      elif ((float(kfold20_40_count[2]) / float(building_total)) * 100.0) < 20.0:
          kfold20_40_list.append(neighbourhood_code)
          kfold20_40_count[0] = kfold20_40_count[0] + class_buildings_neighbourhood[idx]
          kfold20_40_count[1] = kfold20_40_count[1] + non_class_buildings_neighbourhood[idx]
          kfold20_40_count[2] = kfold20_40_count[2] + total_buildings_neighbourhood[idx]
      elif ((float(kfold40_60_count[2]) / float(building_total)) * 100.0) < 20.0:
          kfold40_60_list.append(neighbourhood_code)
          kfold40_60_count[0] = kfold40_60_count[0] + class_buildings_neighbourhood[idx]
          kfold40_60_count[1] = kfold40_60_count[1] + non_class_buildings_neighbourhood[idx]
          kfold40_60_count[2] = kfold40_60_count[2] + total_buildings_neighbourhood[idx]
      elif ((float(kfold60_80_count[2]) / float(building_total)) * 100.0) < 20.0:
          kfold60_80_list.append(neighbourhood_code)
          kfold60_80_count[0] = kfold60_80_count[0] + class_buildings_neighbourhood[idx]
          kfold60_80_count[1] = kfold60_80_count[1] + non_class_buildings_neighbourhood[idx]
          kfold60_80_count[2] = kfold60_80_count[2] + total_buildings_neighbourhood[idx]
      elif ((float(kfold80_100_count[2]) / float(building_total)) * 100.0) < 20.0:
          kfold80_100_list.append(neighbourhood_code)
          kfold80_100_count[0] = kfold80_100_count[0] + class_buildings_neighbourhood[idx]
          kfold80_100_count[1] = kfold80_100_count[1] + non_class_buildings_neighbourhood[idx]
          kfold80_100_count[2] = kfold80_100_count[2] + total_buildings_neighbourhood[idx]

  print("Total number of buildings added " + str(int(building_total + total_buildings_neighbourhood[idx])) +
        " out of " + str(sum(total_buildings_neighbourhood)) + " buildings")

print("------ Kfold neighbourhood lists -----")
print(kfold00_20_list)
print(kfold20_40_list)
print(kfold40_60_list)
print(kfold60_80_list)
print(kfold80_100_list)

print("------ Kfold count building (class, non-class and total) -----")
print(kfold00_20_count)
print(kfold20_40_count)
print(kfold40_60_count)
print(kfold60_80_count)
print(kfold80_100_count)

# Creating training, validation and testing sets per class
if iteration == 0:
  training_neigh = kfold00_20_list + kfold20_40_list + kfold40_60_list
  validation_neigh = kfold60_80_list
  testing_neigh = kfold80_100_list
if iteration == 1:
  training_neigh = kfold20_40_list + kfold40_60_list + kfold60_80_list
  validation_neigh = kfold80_100_list
  testing_neigh = kfold00_20_list
if iteration == 2:
  training_neigh = kfold40_60_list + kfold60_80_list + kfold80_100_list
  validation_neigh = kfold00_20_list
  testing_neigh = kfold20_40_list
if iteration == 3:
  training_neigh = kfold60_80_list + kfold80_100_list + kfold00_20_list
  validation_neigh = kfold20_40_list
  testing_neigh = kfold40_60_list
if iteration == 4:
  training_neigh = kfold80_100_list + kfold00_20_list + kfold20_40_list
  validation_neigh = kfold40_60_list
  testing_neigh = kfold60_80_list

# Set base image directory
base_image_dir = '/home/david/Documents/streetview-master/data'

# Create empty lists for training, validation and testing image filepaths
# One list for class and one for non-class
class_training_images = []
non_class_training_images = []
class_validation_images = []
non_class_validation_images = []
class_testing_images = []
non_class_testing_images = []

# Make list of filepaths of training images for class and non-class
for neigh in training_neigh:
  # Load valid data of specific building class
  rows_neigh = (images_valid['BU_CODE'] == neigh)
  images_neigh = images_valid.loc[rows_neigh, ]
  image_dir = base_image_dir + '/' + neigh[-4:]
  for idx, image in images_neigh.iterrows():
    if len(fov) == 3:
      filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                 + image['pano_id'] + "_" + fov + "_A00.jpg"
      filepaths = [image_dir + "/" + filename]
    if len(fov) == 9:
      filepaths = []
      for fov2 in ['F30','F60','F90']:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                   + image['pano_id'] + "_" + fov2 + "_A00.jpg"
        filepath = image_dir + "/" + filename
        filepaths += [filepath]
    print(filepaths)
    if image[building_class] == 1:
      class_training_images += filepaths
    if image[building_class] == 0:
      non_class_training_images += filepaths

# Make list of filepaths of validation images for class and non-class
for neigh in validation_neigh:
  # Load valid data of specific building class
  rows_neigh = (images_valid['BU_CODE'] == neigh)
  images_neigh = images_valid.loc[rows_neigh,]
  image_dir = base_image_dir + '/' + neigh[-4:]
  for idx, image in images_neigh.iterrows():
    if len(fov) == 3:
      filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                 + image['pano_id'] + "_" + fov + "_A00.jpg"
      filepaths = [image_dir + "/" + filename]
    if len(fov) == 9:
      filepaths = []
      for fov2 in ['F30', 'F60', 'F90']:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                   + image['pano_id'] + "_" + fov2 + "_A00.jpg"
        filepath = image_dir + "/" + filename
        filepaths += [filepath]
    if image[building_class] == 1:
      class_validation_images += filepaths
    if image[building_class] == 0:
      non_class_validation_images += filepaths

# Make list of filepaths of testing images for class and non-class
for neigh in testing_neigh:
  # Load valid data of specific building class
  rows_neigh = (images_valid['BU_CODE'] == neigh)
  images_neigh = images_valid.loc[rows_neigh,]
  image_dir = base_image_dir + '/' + neigh[-4:]
  for idx, image in images_neigh.iterrows():
    if len(fov) == 3:
      filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                 + image['pano_id'] + "_" + fov + "_A00.jpg"
      filepaths = [image_dir + "/" + filename]
    if len(fov) == 9:
      filepaths = []
      for fov2 in ['F30', 'F60', 'F90']:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                   + image['pano_id'] + "_" + fov2 + "_A00.jpg"
        filepath = image_dir + "/" + filename
        filepaths += [filepath]
    if image[building_class] == 1:
      class_testing_images += filepaths
    if image[building_class] == 0:
      non_class_testing_images += filepaths

class_label_name = re.sub(r'[^a-z0-9]+', ' ', building_class.lower())
non_class_label_name = "non-" + class_label_name
result = {}
result[class_label_name] = {
'dir': class_label_name,
'training': class_training_images,
'testing': class_testing_images,
'validation': class_validation_images
}
result[non_class_label_name] = {
'dir': non_class_label_name,
'training': non_class_training_images,
'testing': non_class_testing_images,
'validation': non_class_validation_images
}
print(result)