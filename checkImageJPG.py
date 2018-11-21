#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr
import warnings
import glob
import re
import os
import pandas


def check_image_jpeg(image_dir):
  """Check if jpg files are correct.

  Check if images in image directory have correct JFIF format,
  remove incorrect images and save info of deleted images in dataframe.

  Args:
    image_dir: String path to a folder containing images.
  """
  if not os.path.isdir(image_dir):
    warnings.warn('Warning: Image directory ', + image_dir + ' not found.')
    return None
  image_dir_pattern = image_dir + '/*.jpg'
  image_list = glob.glob(image_dir_pattern)
  for image_path in image_list:
    if imghdr.what(image_path) != 'jpeg': #Check if image has JFIF format by checking first line
      warnings.warn('Warning: ' + image_path + ' does not have correct jpeg format')
      match = re.search("_B([0-9]*)_", image_path) # Get BuildingID out of image file name
      building_id = match.group(1) #Get first group in match
      building_points.loc[building_points.BuildingID == int(building_id), 'valid_jpg'] = 'No'
      building_points.loc[building_points.BuildingID == int(building_id), 'valid'] = 'No'
      os.remove(image_path)
      print('Image has been removed from directory.')

def check_image_exists(base_image_dir):
  """Check if jpg files are correct.

  Check if image exists at location

  """
  for idx, building in building_points.iterrows():
    for fov in ['F30','F60','F90']:
      filename = "N" + building['BU_CODE'] + "_B" + str(building['BuildingID']) + "_P" \
                 + building['pano_id'] + "_" + fov + "_A00.jpg"
      source_info = "/" + building['BU_CODE'][-4:] + "/" + filename
      filepath = base_image_dir + source_info
      if os.path.isfile(filepath) == False:
        building_points.loc[building_points.BuildingID == building['BuildingID'], 'image_exists'] = 'No'
        building_points.loc[building_points.BuildingID == building['BuildingID'], 'valid'] = 'No'
        print("File did not exist: " + filepath)
      else:
        print("File exists: " + filepath)


# Set input and output files
input_file = './input/BuildingPointsValid.csv'
output_file = './input/BuildingPointsValidImages.csv'
base_image_dir = '/home/david/Documents/streetview-master/data'

# Load csv into pandas dataframe
building_points = pandas.read_csv(input_file)

# Create new column in building points to set if image is valid or not
building_points.loc[:, 'valid_jpg'] = 'Yes'
building_points.loc[:, 'image_exists'] = 'Yes'

# Check every repository of images ordered per neighbourhood
for BU_CODE in building_points.BU_CODE.unique():
  image_dir = base_image_dir + '/' + BU_CODE[-4:]
  check_image_jpeg(image_dir = image_dir)

check_image_exists(base_image_dir)

# Write dataframe to csv file
building_points.to_csv(output_file)
