#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import os.path
import random
import re
import sys
import tarfile
import imghdr
import csv

import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# Load data from csv database with all buildings
input_file = "./input/BuildingPointsValidImages.csv"
building_points = pd.read_csv(input_file)

def create_image_lists(image_dir = '/home/david/Documents/streetview-master/data', building_class = 'Residentia', fov = 'F30'):
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
   Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
  if type(building_class) != type('str'):
    tf.logging.error('Building_class variable (' + str(building_class) + ') does not have string data type.')
  if len(fov) != 3 and len(fov) != 9:
    tf.logging.error('Fov variable (' + str(fov) + ') should have length of 3 or 9).')
  # Load valid data of specific building class
  valid_rows = (building_points['valid'] == 'Yes')
  valid_columns = ['ID', 'BuildingID', 'BU_CODE', 'pano_id', building_class]
  images_valid = building_points.loc[valid_rows,valid_columns]

  # Count amount of buildings per neighbourhood
  total_buildings_neighbourhood = images_valid.groupby('BU_CODE')['BU_CODE'].count()
  class_buildings_neighbourhood = images_valid.groupby('BU_CODE')[building_class].sum()
  non_class_buildings_neighbourhood = total_buildings_neighbourhood - class_buildings_neighbourhood

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

  # Creating training, validation and testing sets per class
  training_neigh = kfold00_20_list + kfold20_40_list + kfold40_60_list
  validation_neigh = kfold60_80_list
  testing_neigh = kfold80_100_list
  building_points['category'] = 'invalid'
  building_points['category'].loc[building_points.BU_CODE.isin(training_neigh) & building_points.valid.isin(['Yes'])] = 'training'
  building_points['category'].loc[building_points.BU_CODE.isin(validation_neigh) & building_points.valid.isin(['Yes'])] = 'validation'
  building_points['category'].loc[building_points.BU_CODE.isin(testing_neigh) & building_points.valid.isin(['Yes'])] = 'testing'
  print(building_points.groupby('category').count())


create_image_lists()
building_points.to_csv("./input/BuildingPointsNeighbourhoodCategoriesIteration0.csv")
