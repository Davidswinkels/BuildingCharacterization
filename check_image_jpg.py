#!/usr/bin/env python2
# -*- coding: utf-8 -*-

## Import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr
import os
import warnings
import glob

def check_image_lists(image_dir, logging = True, log_dir = './log', log_file_name = 'errorlog_images.csv'):
  """Check if jpg files are correct.

  Check if images in image directory have correct JFIF format,
  remove incorrect images and optionally save info of deleted images.


  Args:
    image_dir: String path to a folder containing images.
    logging: Binary True/False option if info of deleted images should be saved
    log_dir: String path to folder for logging
    log_file_name: String file name of log
  """
  if not os.path.isdir(image_dir):
    warnings.warn('Warning: Image directory ', + image_dir + ' not found.')
    return None
  image_dir_pattern = image_dir + '/*.jpg'
  image_list = glob.glob(image_dir_pattern)
  error_jpg_list = []
  for image_path in image_list:
    if imghdr.what(image_path) != 'jpeg':
      #os.remove(image_path)
      if logging:
        error_jpg_list.append(image_path)
      warnings.warn('Warning: ' + image_path + ' does not have correct jpeg format')
  if len(error_jpg_list) > 0:
    with open((log_dir + '/' + log_file_name), 'wb') as error_jpg_file:
      wr = csv.writer(error_jpg_file, quoting=csv.QUOTE_ALL)
      wr.writerow(['BuildingID', 'Filepath'])
        for error_jpg_image in error_jpg_list:
          # Regular expression to get BuildingID out of filepath
          match = re.search("_B([0-9]*)_", error_jpg_image)
          print(match.group(1))
          # Write BuildingID and Filepath to csv file
          wr.writerow([match.group(1), error_jpg_image])

base_image_dir = '/home/david/Documents/streetview-master/data_valid_resid_any'
for fov in ['F30', 'F60', 'F90', 'F30_60_90']:
  for image_class in ['residential', 'non_residential']:
    for sub_dir in ['training', 'validation', 'testing']:
      full_image_dir = base_image_dir + '/' + fov + '/' + image_class + '/' + sub_dir
      check_image_lists(image_dir = (full_image_dir),
            logging = True,
            log_dir = '/home/david/PycharmProjects/tensorflow_training/BuildingCharacterization/log',
            log_file_name = 'errorjpg_' + image_class + '_' + fov + '_' + sub_dir + '.csv')