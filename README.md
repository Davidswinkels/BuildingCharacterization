# BuildingClassification
This repository holds scripts to process building classes and predict building functions based on streetview images by finetuning a tensorflow convolutional neural network.

A dataset of buildings, the BAG ([viewer](https://bagviewer.kadaster.nl/lvbag/bag-viewer/index.html#?geometry.x=121736.29375&geometry.y=487599.39169571&zoomlevel=4)|[get as data via WFS](https://geodata.nationaalgeoregister.nl/bag/wfs?request=GetCapabilities), was used to derive all building centroids and functions in Amsterdam.

__concatenate_building_functions.Rmd__: The building functions were concatenated from multiple text entries to multi-label numerical classes. Functions for buildings are residential, meeting, industry, office, accomodation, education, shop and other.

__download_streetview_images.py__: This script still has to be added.

__create_distribution_images.py__: After having the data in the correct format, a random distribution of buildings is appointed to training (60%), validation (20%) and testing (20%) categories. The random distribution is made so that buildings in the same neighbourhood do not get in separate categories. Otherwise training and testing might be biased, because two identical buildings in the same street can both be in training and testing dataset.

__retrain_adj.py__: Re-training a convolutional neural network with premade training, validation and test datasets and to give more outcome statistics. Script was modified fromn [image retraining script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py) by Tensorflow.

__create_backup.py__: A script that makes a back-up of reports/scripts and stores it on a USB-stick.

Feel free to download and use these scripts.

## Document information
__author__ = "David Swinkels"
__github__ = "davidswinkels"
__purpose__ = "Part of MSc thesis Geo-Information Science at Wageningen University"
__status__ = "Production"
