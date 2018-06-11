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
import csv
import collections

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

def create_image_lists(image_dir, building_class, fov , iteration):
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
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
  if type(building_class) != type('str'):
    tf.logging.error('Building_class variable (' + str(building_class) + ') does not have string data type.')
  if len(fov) != 3 and len(fov) != 9:
    tf.logging.error('Fov variable (' + str(fov) + ') should have length of 3 or 9).')
  if type(iteration) != type(0):
    tf.logging.error('Iteration variable (' + str(iteration) + ') does not have integer data type.')
  if iteration < 0 or iteration > 3:
    tf.logging.error('Iteration variable (' + str(iteration) + ') does not have number between 0 and 3 (inclusive).')
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
    image_dir_neigh = image_dir + '/' + neigh[-4:]
    for idx, image in images_neigh.iterrows():
      if len(fov) == 3:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                       + image['pano_id'] + "_" + fov + "_A00.jpg"
        filepaths = [image_dir_neigh + "/" + filename]
      if len(fov) == 9:
        filepaths = []
        for fov2 in ['F30','F60','F90']:
          filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
           + image['pano_id'] + "_" + fov2 + "_A00.jpg"
          filepath = image_dir_neigh + "/" + filename
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
    image_dir_neigh = image_dir + '/' + neigh[-4:]
    for idx, image in images_neigh.iterrows():
      if len(fov) == 3:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                   + image['pano_id'] + "_" + fov + "_A00.jpg"
        filepaths = [image_dir_neigh + "/" + filename]
      if len(fov) == 9:
        filepaths = []
        for fov2 in ['F30', 'F60', 'F90']:
          filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                     + image['pano_id'] + "_" + fov2 + "_A00.jpg"
          filepath = image_dir_neigh + "/" + filename
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
    image_dir_neigh = image_dir + '/' + neigh[-4:]
    for idx, image in images_neigh.iterrows():
      if len(fov) == 3:
        filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                   + image['pano_id'] + "_" + fov + "_A00.jpg"
        filepaths = [image_dir_neigh + "/" + filename]
      if len(fov) == 9:
        filepaths = []
        for fov2 in ['F30', 'F60', 'F90']:
          filename = "N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" \
                     + image['pano_id'] + "_" + fov2 + "_A00.jpg"
          filepath = image_dir_neigh + "/" + filename
          filepaths += [filepath]
      if image[building_class] == 1:
        class_testing_images += filepaths
      if image[building_class] == 0:
        non_class_testing_images += filepaths
  class_label_name = re.sub(r'[^a-z0-9]+', ' ', building_class.lower())
  non_class_label_name = "non_" + class_label_name
  result = collections.OrderedDict()
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
  return result

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist' + label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist' + category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  full_path = category_list[mod_index]
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
  """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    architecture: The name of the model architecture.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Then run it through the recognition network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract(data_url):
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.

  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                    'bytes.')
    print('Extracting file from ', filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  else:
    print('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist ' + image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of which set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    architecture: The name of the model architecture.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
  #label_lists = image_lists[label_name]
  #sub_dir = label_lists['dir']
  #sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  #ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, architecture)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    architecture: The name of the model architecture.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    architecture: The name of the model architecture.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, architecture)
      bottlenecks.append(bottleneck)
      ground_truths.append(label_index)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)
        bottlenecks.append(bottleneck)
        ground_truths.append(label_index)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    bottleneck_tensor_size: How many entries in the bottleneck vector.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [None], name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, ground_truth_tensor)
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def create_model_info(architecture):
  """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
  architecture = architecture.lower()
  is_quantized = False
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True

    if is_quantized:
      data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
      data_url += version_string + '_' + size_string + '_quantized_frozen.tgz'
      bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
      resized_input_tensor_name = 'Placeholder:0'
      model_dir_name = ('mobilenet_v1_' + version_string + '_' + size_string +
                        '_quantized_frozen')
      model_base_name = 'quantized_frozen_graph.pb'

    else:
      data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
      data_url += version_string + '_' + size_string + '_frozen.tgz'
      bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
      resized_input_tensor_name = 'input:0'
      model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
      model_base_name = 'frozen_graph.pb'

    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
      'quantize_layer': is_quantized,
  }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image

def days_hours_minutes_seconds(td):
  return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60

def main(_):

  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare necessary directories that can be used during training
  prepare_file_system()
  # Gather information about the model architecture we'll be using.
  model_info = create_model_info(FLAGS.architecture)
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Set up the pre-trained graph.
  maybe_download_and_extract(model_info['data_url'])
  start_time = datetime.now()  # Save starting time after downloading model
  graph, bottleneck_tensor, resized_image_tensor = (
    create_model_graph(model_info))

  # Give image directory
  image_dir = FLAGS.image_dir

  # Create image lists and check if there are enough classes
  image_lists = create_image_lists(image_dir = image_dir,
      building_class = building_class, fov = fov, iteration=iteration)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('Creating image list unsuccessful -'
                     ' no valid classes were made')
    return -1
  if class_count == 1:
    tf.logging.error('Creating image list unsuccessful' 
                     '- multiple classes are needed for classification')
    return -1

  #Set up naming scheme for output .csv files
  f_class = building_class + '_'
  f_fov = fov + '_'
  f_model = str(FLAGS.architecture) + '_'
  f_iteration = str(iteration)
  f_name = f_class + f_fov + f_model + f_iteration + '.csv'

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    # We'll make sure we've calculated the 'bottleneck' image summaries and
    # cached them on disk.
    cache_bottlenecks(sess, image_lists, image_dir,
                    FLAGS.bottleneck_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor,
                    bottleneck_tensor, FLAGS.architecture)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'], model_info['quantize_layer'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values from the cache stored on disk.
      (train_bottlenecks,train_ground_truth, _) = get_random_cached_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'training',
          FLAGS.bottleneck_dir, image_dir, jpeg_data_tensor,
          decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
          FLAGS.architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        #tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
        #tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        #tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %(datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))

      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name)

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
        sess, image_lists, FLAGS.test_batch_size, 'testing',
        FLAGS.bottleneck_dir, image_dir, jpeg_data_tensor,
        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
        FLAGS.architecture))
    test_accuracy, predictions = sess.run([evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
        ground_truth_input: test_ground_truth})

    # Testing accuracy of predictions with statistics: 
    # overall test accuracy, average accuracy, confusion matrix, kappa stats,
    # precision, recall, computation time, wrongly predicted building ID
    test_ground_truth_pd = pd.Series(test_ground_truth, name = "Actual")
    test_predictions_pd = pd.Series(predictions, name="Predicted")
    conf_matrix = pd.crosstab(test_ground_truth_pd,test_predictions_pd,
        rownames=['Actual'], colnames=['Predicted'], margins=True)

    #tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

    class_accuracy = float(conf_matrix[0][0]) / float(conf_matrix['All'][0]) * 100.0
    non_class_accuracy = float(conf_matrix[1][1]) / float(conf_matrix['All'][1]) * 100.0
    average_accuracy = (class_accuracy + non_class_accuracy) / 2

    # Count number of training, validation and testing images
    class_label_name = re.sub(r'[^a-z0-9]+', ' ', building_class.lower())
    non_class_label_name = "non_" + class_label_name
    train_count = len(image_lists[class_label_name]['training']) +\
                  len(image_lists[non_class_label_name]['training'])
    validation_count = len(image_lists[class_label_name]['validation']) +\
                       len(image_lists[non_class_label_name]['validation'])
    test_count = len(image_lists[class_label_name]['testing']) +\
                 len(image_lists[non_class_label_name]['testing'])

    proport_correct = (float(conf_matrix[0][0]) + float(conf_matrix[1][1])) / (
        float(conf_matrix.at[('All','All')]))
    prob_class = (((float(conf_matrix[0][0]) + float(conf_matrix[1][0])) /
        float(conf_matrix.at[('All','All')]))) * (
        (float(conf_matrix[0][0]) + float(conf_matrix[0][1])) /
        float(conf_matrix.at[('All','All')]))
    prob_non_class = (((float(conf_matrix[0][1]) + float(conf_matrix[1][1])) /
        float(conf_matrix.at[('All','All')]))) * (
        (float(conf_matrix[1][0]) + float(conf_matrix[1][1])) /
        float(conf_matrix.at[('All','All')]))
    prob_all = prob_class + prob_non_class
    kappa = (proport_correct - prob_all) / (1 - prob_all)

    precision = conf_matrix[0][0] / conf_matrix[0]['All']
    recall = conf_matrix[0][0] / conf_matrix['All'][0]

    comp_time = datetime.now() - start_time

    build_pred_error = []
    for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i]:
            match = re.search("_B([0-9]*)_", test_filename)
            build_pred_error.append(match.group(1))

    # Write out confusion matrix, outcome statistics and list of misclassified test images to file
    conf_file_name = '/conf_' + f_name
    conf_matrix.to_csv(FLAGS.log_dir + conf_file_name)

    stats_file_name = '/stats_' + f_name
    with open((FLAGS.log_dir+stats_file_name), 'wb') as stats_file:
        wr = csv.writer(stats_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['train_n', 'validation_n', 'test_n','kappa','precision',
                     'recall','computation_time(seconds)','test_accuracy', 'average_accuracy'])
        row = [train_count,validation_count,test_count, kappa,precision,recall,comp_time.seconds,test_accuracy*100, average_accuracy]
        wr.writerow(row)
    misclass_file_name = '/misclass_' + f_name
    with open((FLAGS.log_dir+misclass_file_name), 'wb') as misclass_file:
        wr = csv.writer(misclass_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['BuildingID'])
        for misclass_image in build_pred_error:
            wr.writerow([misclass_image])
    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i]:
          tf.logging.info('%70s  %s' %
                          (test_filename,
                           list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(sess, graph, FLAGS.output_graph)
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='/home/david/Documents/streetview-master/data',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/david/PycharmProjects/tensorflow_training/BuildingCharacterization/log',
      help='Path to folder of log: confusion matrix, errors, outcome statistics, missclassified images'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--architecture',
      type=str,
      default='inception_v3',
      help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
  FLAGS, unparsed = parser.parse_known_args()

  # Creating input variables
  iterations = [0, 1, 2, 3]
  #building_classes_all = ['Residentia', 'Meeting', 'Healthcare', 'Industry', 'Office','Accommodat', 'Education', 'Sport', 'Shop', 'Other']
  #building_classes = ['Residentia', 'Meeting', 'Industry', 'Office', 'Shop']
  building_classes = ['Meeting', 'Industry', 'Office', 'Shop']
  architectures = ['inception_v3','mobilenet_1.0_224']
  fovs = ['F30', 'F60', 'F90', 'F30_60_90']

  # Looping over CNN models
  for iteration in iterations:
    for building_class in building_classes:
      for architecture in architectures:
        for fov in fovs:
          print("CNN model: ", building_class, fov, architecture, str(iteration))
          FLAGS.architecture = architecture
          tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
