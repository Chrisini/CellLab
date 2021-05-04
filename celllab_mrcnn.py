#!/usr/bin/env python
# coding: utf-8

# Cell Lab
# Cell Lab is a software, that applies instance segmentation on fibroblast cells.

# The base of this project are:
# https://github.com/matterport/Mask_RCNN/tree/master/samples/shapes
# https://github.com/navidyou/Mask-RCNN-implementation-for-cell-nucleus-detection-executable-on-google-colab-

# You can read more about multiple classes here: 
# https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079
# https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07

# Import libraries:
# Requirements regarding requirements.txt ... this needs change!!!!
# https://github.com/matterport/Mask_RCNN/blob/master/requirements.txt

print("")
print("********** ********** START OF CELLLAB MASK RCNN ********** **********")
print("")

cell_debug = False

# ********** PATHS **********
import os
import sys
# root path
ROOT_PATH = os.getcwd()
print('Root:', ROOT_PATH)
# model path
MODEL_PATH = os.path.join(ROOT_PATH, "logs")
print('Model:', MODEL_PATH)
# define image paths
TRAIN_PATH = os.path.join(ROOT_PATH, "named_images_split/train")
VAL_PATH = os.path.join(ROOT_PATH,   "named_images_split/val")
TEST_PATH = os.path.join(ROOT_PATH,  "named_images_split/test")
MASK_PATH = os.path.join(ROOT_PATH,  "named_masks")
MANUAL_MASK_FILE = os.path.join(ROOT_PATH,  "manual_masks/labels_martinahelmlinger_2021-01-30-05-44-23.json")

# ********** mrcnn **********
sys.path.append(ROOT_PATH)
from mrcnn import config
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log
from mrcnn import visualize

class_names = ['BG', 'alive', 'inhib', 'dead', 'fibre']

# ********** .h5 / coco weights **********
import h5py
print('h5py version:', h5py.__version__)
COCO_PATH = os.path.join(ROOT_PATH, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_PATH):
    utils.download_trained_weights(COCO_PATH)
print('COCO h5:', COCO_PATH)

# ********** tensorflow **********
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
#print(tf.python.client.device_lib.list_local_devices())

# ********** keras **********
import keras
print('Keras version:', keras.__version__)
from keras import backend as K

# ********** other packages **********
import pandas as pd
import numpy as np
import scipy
from PIL import Image
import cython
import random
import math
import re
import time
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import cv2
from imgaug import augmenters as iaa
import csv

# ********** matplotlib **********
import matplotlib
import matplotlib.pyplot as plt
def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

import configparser

config = configparser.ConfigParser()
config.read('example.ini')


print("")
print("********** ********** Imports done ********** **********")
print("")

# ********** image and mask IDs + filter **********
train_ids_all = os.listdir(TRAIN_PATH)
val_ids_all = os.listdir(VAL_PATH)
test_ids = os.listdir(TEST_PATH)
mask_ids_all = os.listdir(MASK_PATH)
train_ids, val_ids, mask_ids = [], [], []

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

if cell_debug == True:
  print(train_ids) # prints the train ID's of each directory of an image
  print(val_ids) # prints the train ID's of each directory of an image
  print(test_ids) # prints the test ID's of each directory of an image
  print(mask_ids)

import natsort
test_ids = natsort.natsorted(test_ids)

print("")
print("********** ********** Load image / mask IDs done ********** **********")
print("")

# ********** ********** ********** *********** ********** ********** **********
# ********** ********** ********** CELL CONFIG ********** ********** **********
# ********** ********** ********** *********** ********** ********** **********
class CellConfig(Config):

    NAME = "cells"
    BACKBONE = "resnet50"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = len(class_names)
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32 #400
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = len(train_ids)
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = len(val_ids)

# ********** ********** ********** ********* ********** ********** **********
# ********** ********** ********** INFERENCE ********** ********** **********
# ********** ********** ********** ********* ********** ********** **********
class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

# ********** ********** ********** ************ ********** ********** **********
# ********** ********** ********** CELL DATASET ********** ********** **********
# ********** ********** ********** ************ ********** ********** **********
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
    for index, class_name in enumerate(class_names):
      if index != 0:
        self.add_class("cells", index, class_name)

    if mode == "train":
      for n, image_id in enumerate(train_ids):
        self.add_image("cells", image_id=image_id, path = TRAIN_PATH) # need more here...
        # todo: maybe we need the path for image id rather than the image
        # images were in dirs with an id for each image before ...

    if mode == "val":
      for n, image_id in enumerate(val_ids):
          self.add_image("cells", image_id=image_id, path = VAL_PATH)


    if mode == "test":
      for n, image_id in enumerate(test_ids):
        self.add_image("cells", image_id=image_id, path = TEST_PATH)

  # load image in right shape
  # acutally load one image + resize
  def load_image(self, image_id):

    # this object
    info = self.image_info[image_id]

    try:
      image = imread(info.get("path") + "/" + info.get("id"))
      image = image[0:255, 0:255]
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

    mask = []
    for mask_dict in mask_list:
      for mask_file in mask_dict["mask"]:
        cells.append(mask_dict["name"])
        file = imread(MASK_PATH + '/' + mask_file, as_gray=True).astype(np.bool_)
        # print(file.shape)
        file = file[0:255, 0:255]
        mask.append(file)

    mask = np.stack(mask, axis=-1)
    mask = resize(mask, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

    # Map class names to class IDs.
    class_ids = np.array([self.class_names.index(c) for c in cells])

    # Return mask and according classes
    return mask.astype(np.bool_), class_ids.astype(np.int32)

  def load_coco_mask(self, image_id):

    # this object
    info = self.image_info[image_id]

    print("load manual coco masks")
    from pycocotools.coco import COCO
    annFile=MANUAL_MASK_FILE # .format(dataDir,dataType)

    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    catIds = coco.getCatIds() # catNms=nms# ['alive','inhib', 'dead', 'fibre']);
    imgIds = coco.getImgIds() # catIds=catIds) # ;
    print(catIds)
    print(imgIds)
    print(image_id)
    img = coco.loadImgs(imgIds[image_id])[0] # -1???? index out of range

    ii = os.path.join(TEST_PATH, img["file_name"])
    I = imread(ii)
    #plt.axis('off')
    #plt.imshow(I)
    #plt.show()


    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    cells2 = []
    mask = [] #  = coco.annToMask(anns[0])

    for i in range(len(anns)):
        file = coco.annToMask(anns[i]).astype(np.bool_)
        #print(file.shape)
        #print(file)
        #print(file.type)

        # anns[i]["category_id"]
        entity_id = anns[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]

        cells2.append(entity)#)"dead")
        file = file[0:255, 0:255]
      
        #plt.imshow(file)
        #plt.show()
        mask.append(file)

    mask = np.stack(mask, axis=-1).astype(np.bool_)
    mask = resize(mask, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

    # Map class names to class IDs.
    class_ids = np.array([self.class_names.index(c) for c in cells2])

    # Return mask and according classes
    return mask, class_ids.astype(np.int32)


  def load_coco_mask_2(self, image_id):

    info = self.image_info[image_id]

    #mask = []
    #for annotation in annotations:
    #  coco.annToMask(annotation)

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)

    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)

    #return mask.astype(np.bool_), class_ids.astype(np.int32)



# ********** ********** ********** ********* ********** ********** **********
# ********** ********** ********** FUNCTIONS ********** ********** **********
# ********** ********** ********** ********* ********** ********** **********
def cell_model_setup():
  global config
  config = CellConfig()
  config.display()

def cell_load_datasets():

  global dataset_train 
  dataset_train = CellDataset()
  dataset_train.load_cells("train")
  dataset_train.prepare()

  global dataset_val 
  dataset_val = CellDataset()
  dataset_val.load_cells("val")
  dataset_val.prepare()

  global dataset_test
  dataset_test = CellDataset()
  dataset_test.load_cells("test")
  dataset_test.prepare()

  print("")
  print("********** ********** Val datasets prepared ********** **********")
  print("")

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
  print("********** ********** Train datasets prepared ********** **********")
  print("")


def cell_show_random_samples():
  image_ids = np.random.choice(dataset_train.image_ids, 4)
  # image_ids = [7]

  for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    print(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


def cell_training():

  model = modellib.MaskRCNN(mode="training", 
                            config=config, 
                            model_dir=MODEL_PATH)

  print("")
  print("********** ********** Create 'training' model done ********** **********")
  print("")

  # Load weights from COCO 
  try:
    model.load_weights(COCO_PATH, by_name=True, 
                     exclude=["mrcnn_class_logits", 
                              "mrcnn_bbox_fc", 
                              "mrcnn_bbox", 
                              "mrcnn_mask"])
  except Exception as e:
    print("CellError: couldn't load weights, will try training without pre-trained weights", e)
    print("If OSError, you should delete the .h5 file and download it again (currupted due to e.g. keyboard interrupt).")
    print("If the weights have already been initialised, you should restart the runtime.")

  print("")
  print("********** ********** Load COCO weights done ********** **********")
  print("")

  # Train heads
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
              epochs=100, 
              layers="all")

  print("")
  print("********** ********** Fine tune done ********** **********")
  print("")

def cell_inference(model_path, data, mode, image_id = 8):

  inference_config = InferenceConfig()

  # Recreate the model in inference mode
  global model
  model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_PATH)

  # Load trained weights
  print("Loading weights from ", model_path)
  model.load_weights(model_path, by_name=True)

  if "auto" in mode:
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(data, inference_config,
                                   image_id, use_mini_mask=False)
  elif "manu" in mode:
    
    original_image = dataset_test.load_image(image_id)
    gt_mask, gt_class_id = dataset_test.load_coco_mask(image_id)
    #_idx = np.sum(gt_mask, axis=(0, 1)) > 0
    
    #gt_mask = gt_mask[:, :, _idx]

    gt_bbox = utils.extract_bboxes(gt_mask)
  # Test on a random image
  #original_image, image_meta, gt_class_id, gt_bbox, gt_mask=modellib.load_image_gt(dataset_val, inference_config, 
  #                           image_id, use_mini_mask=False)


  log("original_image", original_image)
  # log("image_meta", image_meta)
  log("gt_class_id", gt_class_id)
  log("gt_bbox", gt_bbox)
  log("gt_mask", gt_mask)

  visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

  results = model.detect([original_image], verbose=1)

  r = results[0]
  visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores']) # , ax=get_ax()


def cell_calculate_metrics(model_path, data, mode):

  inference_config = InferenceConfig()

  image_ids = data.image_ids # np.random.choice(dataset_val.image_ids, 10)
  APs, precisions, recalls, overlaps = [], [], [], []

  for image_id in image_ids:
      # Load image and ground truth data
      if "auto" in mode:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(data, inference_config,
                                   image_id, use_mini_mask=False)
      elif "manu" in mode:
        
        image = dataset_test.load_image(image_id)
        gt_mask, gt_class_id = dataset_test.load_coco_mask(image_id)
        _idx = np.sum(gt_mask, axis=(0, 1)) > 0
        
        gt_mask = gt_mask[:, :, _idx]

        gt_bbox = utils.extract_bboxes(gt_mask)
        

      #molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
      # Run object detection
      results = model.detect([image], verbose=0)
      r = results[0]
      # Compute AP
      AP, precision, recall, overlap =\
          utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                           r["rois"], r["class_ids"], r["scores"], r['masks'])
      APs.append(AP)
      precisions.append(np.mean(precision))
      recalls.append(np.mean(recall))
      overlaps.append(np.mean(overlap))
      
  print("mAP: ", np.mean(APs))

  with open('logs/overview.csv', 'a+') as csvfile:
    cell_writer = csv.writer(csvfile, delimiter=';')
    # cell_writer.writerow(['File', 'mAP'])
    cell_writer.writerow([model_path, mode, np.mean(APs), np.mean(precisions), np.mean(recalls), np.mean(overlaps), len(image_ids)])



# ********** TEST aka images without a ground truth **********
def cell_test(model_path, image_id = 4):

  inference_config = InferenceConfig()

  # Recreate the model in inference mode
  model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_PATH)

  # model_path = model.find_last()
  # Load trained weights
  print("Loading weights from ", model_path)
  model.load_weights(model_path, by_name=True)

  X_test = np.zeros((len(test_ids), inference_config.IMAGE_SHAPE[0], inference_config.IMAGE_SHAPE[1], 3), dtype=np.uint8)
  sizes_test = []
  _test_ids = []

  print('Getting and resizing test images ... ')
  #sys.stdout.flush()
  for n, id_ in enumerate(test_ids):
      _test_ids.append([id_])
      
      img = imread(TEST_PATH + "/" + id_)# [:,:,:3]
      sizes_test.append([img.shape[0], img.shape[1]])
      img = resize(img,
                   (inference_config.IMAGE_SHAPE[0], inference_config.IMAGE_SHAPE[1]), 
                   mode='constant', 
                   preserve_range=True)
      X_test[n] = img


  print("checking a test image with masks ...")
  results = model.detect([X_test[image_id]], verbose=1) # 0 to 5 due to 6 test images

  r = results[0]
  visualize.display_instances(X_test[image_id], 
                              r['rois'], r['masks'], r['class_ids'], 
                              dataset_val.class_names, r['scores']) #, ax=get_ax())"""




# ********** ********** ********** ********** ********** ********** **********
# ********** ********** ********** FUNC CALLS ********** ********** **********
# ********** ********** ********** ********** ********** ********** **********

# Get path to saved weights
load_model_path = os.path.join(MODEL_PATH, "cells20210503T0938/mask_rcnn_cells_0018.h5")


def main_training():
  cell_model_setup()
  cell_load_datasets()
  cell_show_random_samples()
  cell_training()

def main_inference_auto():
  cell_model_setup()
  cell_load_datasets()
  cell_inference(model_path = load_model_path, data = dataset_val, mode = "automatical masks", image_id = 2) # 8
  # for path in paths: walk dir .h5
  cell_calculate_metrics(model_path = load_model_path, data = dataset_val, mode = "automatical masks")

def main_inference_manual():
  cell_model_setup()
  cell_load_datasets()
  cell_inference(model_path = load_model_path, data = dataset_test, mode = "manual masks", image_id = 5) # 2
  # for path in paths: walk dir .h5
  cell_calculate_metrics(model_path = load_model_path, data = dataset_test, mode = "manual masks")

def main_testing():
  cell_model_setup()
  cell_test(model_path = load_model_path, image_id = 2)


#main_training()
#main_inference_auto()
main_inference_manual()
#main_testing()



#    no_ids = [ t[idx] for idx in range(0, len(t), 2)]
"""
skimage.draw.polygon(p['all_points_y'], p['all_points_x'])



# class of each cell in image
cells = []

# get the mask ids for each class (alive, inhib, dead, fibre)
mask_list = [{'name': 'alive'}, {'name': 'inhib'}, {'name': 'dead'}, {'name': 'fibre'}] # 1, 2, 3, 4

all_masks_for_this_image = [i for i in mask_ids if info.get("id") in i]
for mask_dict in mask_list:
  mask_dict["mask"] = [i for i in all_masks_for_this_image if mask_dict["name"] in i]

mask = []
for mask_dict in mask_list:
  for mask_file in mask_dict["mask"]:
    cells.append(mask_dict["name"])
    file = imread(MASK_PATH + '/' + mask_file, as_gray=True).astype(np.bool_)
    file = file[0:255, 0:255]
    mask.append(file)
"""
