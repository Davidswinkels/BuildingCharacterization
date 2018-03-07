#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

# Load data from csv database with all buildings
input_file = "./input/BuildingPointsValidImages.csv"
building_points = pd.read_csv(input_file)

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def create_image_lists(image_dir = '/home/david/Documents/streetview-master/data',
                       building_class = 'Residentia', fov = 'F30' , iteration = 0 ):
  """Builds a list of training, validation and testing images from the file system.
  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.
  Args:
    image_dir: String path to a folder containing subfolders of images.
    building_class: String name of building class (one of 10 building classes:
        'Residentia', 'Meeting', 'Healthcare', 'Industry', 'Office',
        'Accommodat', 'Education', 'Sport', 'Shop', 'Other')
    fov: String name of field of view (one of four fovs: 'F30', 'F60', 'F90', 'F30_60_90')
    iteration: Numeric value between 0 and 3 (inclusive)
   Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  global validation_neigh
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  if type(building_class) != type('str'):
    tf.logging.error('building_class variable (', str(building_class),') does not have string data type')
  if len(fov) != 3 or len(fov) != 9:
    tf.logging.error('fov variable (', str(fov), ') does not have correct length of characters (should be 3 or 9)')
  if type(iteration) != type(0):
    tf.logging.error('iteration variable (', str(iteration),') does not have integer data type')
  if iteration < 0 or iteration > 3:
    tf.logging.error('iteration variable (', str(iteration), ') does not have number between 0 and 3 (inclusive)')
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
    validation_neigh = kfold00_20_list
    testing_neigh = kfold80_100_list
  if iteration == 2:
    training_neigh = kfold40_60_list + kfold60_80_list + kfold00_20_list
    validation_neigh = kfold20_40_list
    testing_neigh = kfold80_100_list
  if iteration == 3:
    training_neigh = kfold60_80_list + kfold00_20_list + kfold20_40_list
    validation_neigh = kfold40_60_list
    testing_neigh = kfold80_100_list


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
    images_neigh = images_valid.loc[rows_neigh,]
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
  non_class_label_name = "non_" + class_label_name

  # Make directory to store bottlenecks
  class_bottleneck_dir = base_image_dir + "/bottleneck/" + class_label_name
  non_class_bottleneck_dir = base_image_dir + "/bottleneck/" + non_class_label_name
  ensure_dir_exists(class_bottleneck_dir)
  ensure_dir_exists(non_class_bottleneck_dir)

  # Combine lists in dictionary and return result of combined lists
  result = {}
  result[class_label_name] = {
    'dir': class_bottleneck_dir,
    'training': class_training_images,
    'testing': class_testing_images,
    'validation': class_validation_images
  }
  result[non_class_label_name] = {
    'dir': non_class_bottleneck_dir,
    'training': non_class_training_images,
    'testing': non_class_testing_images,
    'validation': non_class_validation_images
  }
  return result


image_lists = create_image_lists()

print(image_lists.keys())
print(image_lists['residentia'].keys())
print(image_lists['non_residentia'].keys())