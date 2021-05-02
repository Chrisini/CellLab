#!/usr/bin/env python
# coding: utf-8

# ## Cell Lab
# Cell Lab is a software, that applies instance segmentation on fibroblast cells.
# 
# The base of this project are:
# 
# https://github.com/matterport/Mask_RCNN/tree/master/samples/shapes
# 
# https://github.com/navidyou/Mask-RCNN-implementation-for-cell-nucleus-detection-executable-on-google-colab-

# ## Import libraries
# 
# Requirements regarding requirements.txt:
# https://github.com/matterport/Mask_RCNN/blob/master/requirements.txt

print("")
print("********** ********** START OF CELLLAB MASK RCNN ********** **********")
print("")

import os
import sys



# ********** PATHS **********

ROOT_PATH = os.getcwd()
print('Root:', ROOT_PATH)

sys.path.append(ROOT_PATH)

MODEL_PATH = os.path.join(ROOT_PATH, "logs")
print('Model:', MODEL_PATH)


# define image paths
TRAIN_PATH = os.path.join(ROOT_PATH, "named_images_split/train")
VAL_PATH = os.path.join(ROOT_PATH,   "named_images_split/val")
TEST_PATH = os.path.join(ROOT_PATH,  "named_images_split/test")
MASK_PATH = os.path.join(ROOT_PATH,  "named_masks")



# Import Mask RCNN
  # To find local version of the library
from mrcnn import config
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log
from mrcnn import visualize

COCO_PATH = os.path.join(ROOT_PATH, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_PATH):
    utils.download_trained_weights(COCO_PATH)
print('COCO h5:', COCO_PATH)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


import ctypes

# cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin\\"
# to_load = ["cudnn64_7.dll"]


# %tensorflow_version 1.x
import tensorflow as tf
print('TensorFlow version:', tf.__version__)

# !pip install keras==2.0.8
import keras
print('Keras version:', keras.__version__)
from keras import backend as K


import pandas as pd
import numpy as np
import scipy
from PIL import Image
import cython

import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import cv2
import h5py
print('h5py version:', h5py.__version__)
from imgaug import augmenters as iaa

# import IPython

import random
import math
import re
import time

print("")
print("********** ********** Imports done ********** **********")
print("")

# get image and mask IDs
train_ids_all = os.listdir(TRAIN_PATH)
val_ids_all = os.listdir(VAL_PATH)
test_ids = os.listdir(TEST_PATH)
mask_ids_all = os.listdir(MASK_PATH)



train_ids = []
val_ids = []
mask_ids = []

for mask_id in mask_ids_all:
  if "dead" in mask_id or "inhib" in mask_id:
    mask_ids.append(mask_id)


for mask_id in mask_ids:
  if "_0_" in mask_id:

    for train_id in train_ids_all:
      if train_id in mask_id:
        train_ids.append(train_id)
        
    for val_id in val_ids_all:
      if val_id in mask_id:
        val_ids.append(val_id)


# train_ids.remove("cell186.jpg")


#print(train_ids) # prints the train ID's of each directory of an image
#print(val_ids) # prints the train ID's of each directory of an image
#print(test_ids) # prints the test ID's of each directory of an image
#print(mask_ids)


# In[5]:


class CellConfig(Config):

    NAME = "cells"
    BACKBONE = "resnet50"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 cells

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = CellConfig()
config.display()


# ## Classes
# 
# Background(0), alive(1), inhib(2), dead(3), fibre(4)
# 
# when changing classes here, also change NUM_CLASSES constant

# In[6]:


class_names = ['BG', 'alive', 'inhib', 'dead', 'fibre'] # fibre

# Example
# class_names.index('round')


# ## Data

# ###  Data import

# ## Cell Dataset
# 
# You can read more about multiple classes here: 
# 
# https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079
# 
# https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07

# In[7]:


class CellDataset(utils.Dataset):

  def load_class_for_image(self, image_id):
    # not used

    cell_class = []
    
    for mask_id in mask_ids:
      if "cell" in mask_id: # image_id (testing!!!!!!!)
        if "alive" in mask_id:
          cell_class.append("alive")
        elif "inhib" in mask_id:
          cell_class.append("inhib")
        elif "dead" in mask_id:
          cell_class.append("dead")
        elif "fibre" in mask_id:
          cell_class.append("fibre")

    return cell_class

  def load_cells(self, mode): # get here a classes variable (with all cell classes??)

    # why is this here??? we need 4 classes for each image??
    # maybe don't need this in general
    self.add_class("cells", 1, "alive")
    self.add_class("cells", 2, "inhib")
    self.add_class("cells", 3, "dead")
    self.add_class("cells", 4, "fibre")

    if mode == "train":
      for n, image_id in enumerate(train_ids):
        self.add_image("cells", image_id=image_id, path = TRAIN_PATH) # need more here...
        # todo: maybe we need the path for image id rather than the image
        # images were in dirs with an id for each image before ...

    if mode == "val":
      for n, image_id in enumerate(val_ids):
          self.add_image("cells", image_id=image_id, path = VAL_PATH)

  # load image in right shape
  # acutally load one image + resize
  def load_image(self, image_id):

    # this object
    info = self.image_info[image_id]

    try:
      image = imread(info.get("path") + "/" + info.get("id"))
      image = resize(image, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range = True)
    except Exception as e:
      print("file read error", e)
      image = np.zeros([config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],dtype=np.uint8)
    
    return image

  # what???? I think get the cell's stage (dead, inhib, alive, fibre)
  # return the cell shape data of image
  def image_refernce(self, image_id):

    # this object
    info = self.image_info[image_id]
    if info["source"] == "cells":
      return info["cells"]
    else:
      super(self.__class__).image_reference(self, image_id)

  def load_mask(self, image_id):

    # this object
    info = self.image_info[image_id]

    # class of each cell in image
    cells = []

    # get the mask ids for each class (alive, inhib, dead, fibre)
    mask_list = [{'name': 'alive'}, {'name': 'inhib'}, {'name': 'dead'}, {'name': 'fibre'}]
    all_masks_for_this_image = [i for i in mask_ids if info.get("id") in i]
    for mask_dict in mask_list:
      mask_dict["mask"] = [i for i in all_masks_for_this_image if mask_dict["name"] in i]
    
    # create combined mask (showing all cells)
    mask = []
    for mask_dict in mask_list:
      for mask_file in mask_dict["mask"]:
        cells.append(mask_dict["name"])
        file = imread(MASK_PATH + '/' + mask_file, as_gray=True).astype(np.bool)
        mask.append(file)

    mask = np.stack(mask, axis=-1)
    mask = resize(mask, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

    # Map class names to class IDs.
    class_ids = np.array([self.class_names.index(s) for s in cells])

    # Return mask and according classes
    return mask.astype(np.bool), class_ids.astype(np.int32)


# ### Create training and validation dataset

# In[8]:


dataset_train = CellDataset()
dataset_train.load_cells("train")
dataset_train.prepare()

#dataset_val = ShapesDataset()
dataset_val = CellDataset()
dataset_val.load_cells("val")
dataset_val.prepare()

augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

print("")
print("********** ********** Datasets prepared ********** **********")
print("")
# ### Load random images

# In[23]:

"""
image_ids = np.random.choice(dataset_train.image_ids, 4)
# image_ids = [7]

for image_id in image_ids:
  image = dataset_train.load_image(image_id)
  print(image_id)
  mask, class_ids = dataset_train.load_mask(image_id)
  visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
"""

# ## Architecture
# The architecture used is a Mask R-CNN

# In[9]:



model = modellib.MaskRCNN(mode="training", 
                          config=config, 
                          model_dir=MODEL_PATH)

print("")
print("********** ********** Create model done ********** **********")
print("")
# ### Load weights from COCO
# this might throw an error if the weights have already been initialised. Then just restart runtime.
# 
# If OSError: Unable to open file (truncated file: eof = 64028672, sblock->base_addr = 0, stored_eof = 257557808) error raises, you should delete the .h5 file and download it new (currupted due to e.g. keyboard interrupt)

# In[10]:

#"""
model.load_weights(COCO_PATH, by_name=True, 
                   exclude=["mrcnn_class_logits", 
                            "mrcnn_bbox_fc", 
                            "mrcnn_bbox", 
                            "mrcnn_mask"])

print("")
print("********** ********** Load COCO weights done ********** **********")
print("")

# ### Training | Transfer Learning

# In[11]:


model.train(dataset_train, dataset_val, 
            learning_rate = config.LEARNING_RATE, 
            epochs = 3, 
            layers = 'heads')


print("")
print("********** ********** Train heads done ********** **********")
print("")

# Fine tune
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10, 
            layers="all")

print("")
print("********** ********** Fine tune done ********** **********")
print("")

# ## Save Final Weights
# Maybe useless, since I should probably do early stopping

# In[ ]:


model_path = os.path.join(MODEL_PATH, "mask_rcnn_cells_final.h5")
model.keras_model.save_weights(model_path)

print("")
print("Final weights saved")
print("")

# # Detection

# In[ ]:
#"""

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_PATH)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = os.path.join(MODEL_PATH, "mask_rcnn_cells_final.h5")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# # Test on a validation image ... with ground truth? ...

# Ground Truth

# In[ ]:


# Test on a random image
image_id = 8 # random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask=modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))


# Original image with predictions

# In[ ]:


results = model.detect([original_image], verbose=1)

r = results[0]
print(r)
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores']) # , ax=get_ax()


# ## Evaluation

# In[ ]:


"""
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
"""


# ## Test | Final evaluation

# In[ ]:


X_test = np.zeros((len(test_ids), config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype=np.uint8)
sizes_test = []
_test_ids = []

print('Getting and resizing test images ... ')
#sys.stdout.flush()
for n, id_ in enumerate(test_ids):
    _test_ids.append([id_])
    
    img = imread(TEST_PATH + "/" + id_)# [:,:,:3]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img,
                 (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                 mode='constant', 
                 preserve_range=True)
    X_test[n] = img


# In[ ]:


print("checking a test image with masks ...")
results = model.detect([X_test[3]], verbose=1) # 0 to 5 due to 6 test images

r = results[0]
visualize.display_instances(X_test[3], 
                            r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores']) #, ax=get_ax())


