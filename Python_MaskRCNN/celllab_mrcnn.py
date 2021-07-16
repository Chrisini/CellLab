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

print("********** ********** ************************** ********** **********")
print("********** ********** START OF CELLLAB MASK RCNN ********** **********")
print("********** ********** ************************** ********** **********")

cell_debug = False
cell_show_masks = True

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
TRAIN_PATH =  os.path.join(ROOT_PATH, "named_images_split/train")
VAL_PATH =    os.path.join(ROOT_PATH,   "named_images_split/val")
TEST_PATH =   os.path.join(ROOT_PATH,  "named_images_split/test")
MASK_PATH =   os.path.join(ROOT_PATH,  "named_masks")
# manual annotator masks
MANUAL_MASK_FILE_A1 = os.path.join(ROOT_PATH,  "manual_masks/labels_M.json")
MANUAL_MASK_FILE_A2 = os.path.join(ROOT_PATH,  "manual_masks/labels_L.json")
MANUAL_MASK_FILE_A3 = os.path.join(ROOT_PATH,  "manual_masks/labels_G.json")

# ********** mrcnn **********
sys.path.append(ROOT_PATH)
from mrcnn import config
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log
from mrcnn import visualize

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class_names_all = ['BG', 'inhib', 'dead']
#class_names_all = ['BG', 'alive', 'inhib', 'dead', 'fibre']

# ********** .h5 / coco weights **********
import h5py
print('h5py version:', h5py.__version__)
COCO_PATH = os.path.join(ROOT_PATH, "mask_rcnn_coco.h5") 
# todo – check if h5 is only for resnet-101?
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
val_ids_all   = os.listdir(VAL_PATH)
test_ids      = os.listdir(TEST_PATH)
mask_ids_all  = [] # os.listdir(MASK_PATH)

for root, subdirds, files in os.walk(MASK_PATH):
	for file in files:
		m = root + "/" + file
		if "morph" not in m:
			mask_ids_all.append(m)

train_ids, val_ids, mask_ids = [], [], []

# only consider dead and inhibited cells
for mask_id in mask_ids_all:
  #if "dead" in mask_id or "inhib" in mask_id:
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
    NUM_CLASSES = len(class_names_all)
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 400 #400
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = len(train_ids)
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = len(val_ids)

    EPOCHS_HEAD = 5
    EPOCHS_FINE = 100

    
    # MINI_MASK_SHAPE = (96,96)

# ********** ********** ********** ********* ********** ********** **********
# ********** ********** ********** INFERENCE ********** ********** **********
# ********** ********** ********** ********* ********** ********** **********
class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

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
          print("fibre")

    return cell_class

  def load_cells(self, mode): # get here a classes variable (with all cell classes??)

    # why is this here??? we need 4 classes for each image??
    # maybe don't need this in general
    for index, class_name in enumerate(class_names_all):
      if index != 0:
        self.add_class("cells", index, class_name)

    if mode == "train":
      for n, image_id in enumerate(train_ids):
        self.add_image("cells", image_id=image_id, path = TRAIN_PATH) # need more here...
        # todo: maybe we need the path for image id rather than the image
        # images were in dirs with an id for each image before ...
    print(len(train_ids))

    if mode == "val":
      for n, image_id in enumerate(val_ids):
          self.add_image("cells", image_id=image_id, path = VAL_PATH)
    print(len(val_ids))

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
      #image = image[0:200, 0:200]
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
    #mask_list = [{'name': 'alive'}, {'name': 'inhib'}, {'name': 'dead'}, {'name': 'fibre'}]
    mask_list = [{'name': 'inhib'}, {'name': 'dead'}]

    #mask_list = [{'name': 'inhib'}, {'name': 'dead'}]
    all_masks_for_this_image = [i for i in mask_ids if info.get("id") in i]
    for mask_dict in mask_list:
      mask_dict["mask"] = [i for i in all_masks_for_this_image if mask_dict["name"] in i]

    mask = []
    for mask_dict in mask_list:
      for mask_file in mask_dict["mask"]:
        cells.append(mask_dict["name"])
        file = imread(mask_file, as_gray=True).astype(np.bool_) #MASK_PATH + '/' +
        # print(file.shape)
        #file = file[0:200, 0:200]
        mask.append(file)

    if len(mask) == 0:
      print("no elements found in load mask")
      mask = np.zeros([config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],3],dtype=np.uint8).astype(np.bool_)
    
    mask = np.stack(mask, axis=-1)
    
    mask = resize(mask, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

    # Map class names to class IDs.
    class_ids = np.array([class_names_all.index(c) for c in cells])

    # Return mask and according classes
    return mask.astype(np.bool_), class_ids.astype(np.int32)
  
  def annToRLE(self, ann, height, width):
      """
      Convert annotation which can be polygons, uncompressed RLE to RLE.
      :return: binary mask (numpy 2D array)
      """
      segm = ann['segmentation']
      if isinstance(segm, list):
          # polygon -- a single object might consist of multiple parts
          # we merge all parts into one mask rle code
          rles = maskUtils.frPyObjects(segm, height, width)
          rle = maskUtils.merge(rles)
      elif isinstance(segm['counts'], list):
          # uncompressed RLE
          rle = maskUtils.frPyObjects(segm, height, width)
      else:
          # rle
          rle = ann['segmentation']
      return rle
  
  def annToMask(self, ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = self.annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

  def load_coco_mask(self, image_id, coco_path):

    # this object
    info = self.image_info[image_id]

    print("load manual coco masks")
    
    annFile=coco_path # .format(dataDir,dataType)

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
        if file.max() < 1:
            continue
        #print(file.shape)
        #print(file)
        #print(file.type)

        # anns[i]["category_id"]
        entity_id = anns[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]

        cells2.append(entity)#)"dead")
        # file = file[0:200, 0:200]
      
        #plt.imshow(file)
        #plt.show()
        mask.append(file)

    if len(mask) == 0:
      print("no elements found")
      mask = np.zeros([config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],3],dtype=np.uint8).astype(np.bool_)
    
    mask = np.stack(mask, axis=-1)      
      
    mask = resize(mask, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

    # Map class names to class IDs.
    class_ids = np.array([class_names_all.index(c) for c in cells2])

    # Return mask and according classes
    return mask.astype(np.bool_), class_ids.astype(np.int32)


# ********** ********** ********** ****************** ********** ********** **********
# ********** ********** ********** LOAD SEVERAL MASKS ********** ********** **********
# ********** ********** ********** ****************** ********** ********** **********
class MaskLoader():

  def __init__(self, dataset):
    self.dataset = dataset


  def load_mm(self, mask_file, image_id):
    original_image = self.dataset.load_image(image_id)
    mask, classid = self.dataset.load_coco_mask(image_id, mask_file)
    bbox = utils.extract_bboxes(mask)

    log("original_image", original_image)
    log("class_id", classid)
    log("bbox", bbox)
    log("mask", mask)

    if cell_show_masks == True:
      helper = TestSetHelper()
      helper.display_instances(original_image, bbox, mask, classid, class_names_all, figsize=(8, 8))

    return original_image, mask, classid, bbox

  def load_am(self, image_id):
    original_image = self.dataset.load_image(image_id)
    mask, classid = self.dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)

    log("original_image", original_image)
    log("class_id", classid)
    try:
        log("bbox", gt_bbox)
        log("mask", gt_mask)
    except Exception as e:
        print(e)
    

    if cell_show_masks == True:
      helper = TestSetHelper()
      helper.display_instances(original_image, bbox, mask, classid, class_names_all, figsize=(8, 8))

    return original_image, mask, classid, bbox

  def load_cm(self, weight_path, image_id):

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_PATH)

    # Load trained weights
    print("Loading weights from ", weight_path)
    model.load_weights(weight_path, by_name=True)

    original_image = self.dataset.load_image(image_id)
    results = model.detect([original_image], verbose=1)
    r = results[0]

    if cell_show_masks == True:
      helper = TestSetHelper()
      helper.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], class_names_all, r['scores'])

    # original_image, class_id, bbox, mask
    return original_image, r['masks'], r['class_ids'], r['rois'], r['scores']
            

# ********** ********** ********** ************** ********** ********** **********
# ********** ********** ********** TESTSET HELPER ********** ********** **********
# ********** ********** ********** ************** ********** ********** **********
class TestSetHelper():

  def cell_show_random_samples(self, dataset_train):
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    print("here")
    # image_ids = [7]

    for image_id in image_ids:
      image = dataset_train.load_image(image_id)
      print(image_id)
      mask, class_ids = dataset_train.load_mask(image_id)
      if cell_show_masks == True:
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
        
        
  def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    self.display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")
    
  def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
    
  def display_instances(self, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    #ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):

        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "" # {} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1+((x2-x1)/4), y1+((y2-y1)/2), caption,
                color='w', size=10, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color, 0.15)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8).astype(np.bool_)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

  def display_differences(self, image,
                          gt_box, gt_class_id, gt_mask,
                          pred_box, pred_class_id, pred_score, pred_mask,
                          class_names, title="", ax=None,
                          show_mask=True, show_box=True,
                          iou_threshold=0.5, score_threshold=0.5):
      """Display ground truth and prediction instances on the same image."""
      # Match predictions to ground truth
      gt_match, pred_match, overlaps = utils.compute_matches(
          gt_box, gt_class_id, gt_mask,
          pred_box, pred_class_id, pred_score, pred_mask,
          iou_threshold=iou_threshold, score_threshold=score_threshold)
      
      #print(gt_match)
      #print(pred_match)
      
      # Ground truth = green. Predictions = red
      colors = [(1, 1, 0, 1)] * len(gt_match)\
             + [(0, 1, 1, 1)] * len(pred_match)
      # Concatenate GT and predictions
      class_ids = np.concatenate([gt_class_id, pred_class_id])
      scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
      boxes = np.concatenate([gt_box, pred_box])
      masks = np.concatenate([gt_mask, pred_mask], axis=-1)

      
      captions = []

      captions = ["" for m in gt_match] + ["{:.2f}".format(
      (overlaps[i, int(pred_match[i])]
          if pred_match[i] > -1 else overlaps[i].max()))
          for i in range(len(pred_match))]

      print(len(captions))
      #print(captions)

      if cell_show_masks == True:
        self.display_instances(
            image,
            boxes, masks, class_ids,
            class_names, None, ax=ax,
            show_bbox=False, show_mask=show_mask,
            colors=colors, captions=captions,
            title=title)    
        
  def remove_empty(self, bbox, classid, mask): # bbox1 or bbox2
    # CLEAN DATASET - REMOVE EMPTY
    keep_index = np.where(~np.all(bbox == 0, axis=1)) # = use
    delete_index = np.where(np.all(bbox == 0, axis=1)) # = delete wrong ...

    print(bbox.shape)
    bbox = bbox[keep_index]
    classid = classid[keep_index]
    mask = np.delete(mask, delete_index, axis=2)
    return bbox, classid, mask          

  

# ********** ********** ********** ********* ********** ********** **********
# ********** ********** ********** FUNCTIONS ********** ********** **********
# ********** ********** ********** ********* ********** ********** **********
def cell_model_setup():

  global config
  config = CellConfig()
  config.display()

def cell_load_datasets():


  global dataset_train 
  global dataset_val
  global dataset_test

  dataset_train = CellDataset()
  dataset_train.load_cells("train")
  dataset_train.prepare()

  dataset_val = CellDataset()
  dataset_val.load_cells("val")
  dataset_val.prepare()

  dataset_test = CellDataset()
  dataset_test.load_cells("test")
  dataset_test.prepare()

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


# ********** ********** ********** ****************** ********** ********** **********
# ********** ********** ********** TRAINING / WITH GT ********** ********** **********
# ********** ********** ********** ****************** ********** ********** **********
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
            epochs = config.EPOCHS_HEAD, 
            layers = 'heads')

  print("")
  print("********** ********** Train heads done ********** **********")
  print("")

  # Fine tune
  model.train(dataset_train, dataset_val, 
              learning_rate=config.LEARNING_RATE / 10,
              epochs = config.EPOCHS_FINE, 
              layers = "all")

  print("")
  print("********** ********** Fine tune done ********** **********")
  print("")


def cell_compare_testset(mode, compare_1, compare_2, dataset):
    
    print("")
    print("********** ********** Compare Testset ********** **********")
    print(mode)
    print("")

    inference_config = InferenceConfig()

    image_ids = dataset.image_ids # np.random.choice(dataset_val.image_ids, 10)
    APs, precisions, recalls, overlaps = [], [], [], []

    loader = MaskLoader(dataset_test)

    print("Image IDs:", image_ids)
    #image_ids = np.delete(image_ids, 0)
    #image_ids = np.delete(image_ids, 2)

    for image_id in image_ids:
      if "mm1_mm2" in mode or "mm1_mm3" in mode:
        image1, mask1, classid1, bbox1 = loader.load_mm(compare_1, image_id) # file
        _, mask2, classid2, bbox2 = loader.load_mm(compare_2, image_id) # file
        #score2 = np.arange(0, 1, 1/len(classid2))
        numbers = np.arange(0,len(classid2),1)
        score2 = (numbers-np.min(numbers))/(np.max(numbers)-np.min(numbers))
      elif "mm1_am" in mode:
        image1, mask1, classid1, bbox1 = loader.load_mm(compare_1, image_id) # file
        _, mask2, classid2, bbox2 = loader.load_am(image_id) # data
        score2 = np.arange(0, 1, 1/len(classid2))
      elif "mm1_cm" in mode:
        image1, mask1, classid1, bbox1 = loader.load_mm(compare_1, image_id) # file
        _, mask2, classid2, bbox2, score2 = loader.load_cm(compare_2, image_id) # weight_path
      elif "am_cm" in mode:
        image1, mask1, classid1, bbox1 = loader.load_am(image_id) # file
        _, mask2, classid2, bbox2, score2 = loader.load_cm(compare_2, image_id) # weight_path

      
      print("A: Bounding box 1 shape", bbox1.shape)
      # clean dataset
      helper = TestSetHelper()
      bbox1, classid1, mask1 = helper.remove_empty(bbox1, classid1, mask1)
      bbox2, classid2, mask2 = helper.remove_empty(bbox2, classid2, mask2)
      print("B: Bounding box 1 shape", bbox1.shape)

      # put second into correct order
      indices = np.argsort(score2)[::-1]
      # print(indices)
      bbox2 = bbox2[indices]
      classid2 = classid2[indices]
      score2 = score2[indices]
      mask2 = mask2[..., indices]

      # Compute AP
      AP, precision, recall, overlap = utils.compute_ap(bbox1, classid1, mask1,
                             bbox2, classid2, score2, mask2)
      
      print(precision)        
      APs.append(AP)
      precisions.append(np.mean(precision)) # mean precision of one image??? do I get more than one here??? todo
      recalls.append(np.mean(recall))
      overlaps.append(np.mean(overlap))
        
      helper.display_differences(image1,
                            bbox1, classid1, mask1,
                            bbox2, classid2, score2, mask2,
                            class_names = class_names_all) #  dataset.class_names) # class_names_all


    # final outcome
    print("mAP: " + str(np.mean(APs)) + " | precision: " + str(np.mean(precisions)) + " | recall: " + str(np.mean(recalls)) + " | overlaps: " + str(np.mean(overlaps)))
    with open('logs/overview.csv', 'a+') as csvfile:
      cell_writer = csv.writer(csvfile, delimiter=';')
      # cell_writer.writerow(['File', 'mAP'])
      cell_writer.writerow([compare_1, compare_2, mode, np.mean(APs), np.mean(precisions), np.mean(recalls), np.mean(overlaps), len(image_ids)])


def cell_use(weight_path, image_id, dataset):
  inference_config = InferenceConfig()
  loader = MaskLoader(dataset)
  loader.load_cm(weight_path, image_id)



# ********** ********** ********** ********** ********** ********** **********
# ********** ********** ********** FUNC CALLS ********** ********** **********
# ********** ********** ********** ********** ********** ********** **********

def main_training():
  print("")
  print("********** ********** Start training ********** **********")
  print("")
  cell_model_setup()
  cell_load_datasets()
  helper = TestSetHelper()
  helper.cell_show_random_samples(dataset_train)
  cell_training()
  print("")
  print("********** ********** Finish training ********** **********")
  print("")

def main_compare():
  print("")
  print("********** ********** Start compare results ********** **********")
  print("")
  cell_model_setup()
  cell_load_datasets()

  # Get path to saved weights
  # good model: load_model_path = os.path.join(MODEL_PATH, "cells20210503T0938/mask_rcnn_cells_0018.h5")
  # weight_path = os.path.join(MODEL_PATH, "cells20210503T0938/mask_rcnn_cells_0018.h5")
  weight_path = os.path.join(MODEL_PATH, "mask_rcnn_cells_0065.h5")

  compare_list = []
  #compare_list.append({"mode" : "mm1_mm2",  "compare_1" : MANUAL_MASK_FILE_A1, "compare_2" : MANUAL_MASK_FILE_A2})
  #compare_list.append({"mode" : "mm1_mm2",  "compare_1" : MANUAL_MASK_FILE_A1, "compare_2" : MANUAL_MASK_FILE_A3})
  #compare_list.append({"mode" : "mm1_am",   "compare_1" : MANUAL_MASK_FILE_A1, "compare_2" : None})
  #compare_list.append({"mode" : "mm1_cm",   "compare_1" : MANUAL_MASK_FILE_A1, "compare_2" : weight_path})
  #compare_list.append({"mode" : "am_cm",    "compare_1" : None,                "compare_2" : weight_path})
  
  for entry in compare_list:
    cell_compare_testset(entry["mode"], entry["compare_1"], entry["compare_2"], dataset_test)
 
  print("")
  print("********** ********** Finish prediction compare results ********** **********")
  print("")

def main_use():
  print("")
  print("********** ********** Start use prediction ********** **********")
  print("")
  cell_model_setup()
  cell_load_datasets()
  weight_path = os.path.join(MODEL_PATH, "cells20210503T0938/mask_rcnn_cells_0018.h5")
  image_id = 0
  cell_use(weight_path, image_id, dataset_test)
  print("")
  print("********** ********** Finish use prediction ********** **********")
  print("")


def main_plot_curves():
    
    #loss_train = pd.read_csv("logs/run-cells_Fedora-tag-loss.csv", delimiter=",", usecols=["Value"])
    #loss_val = pd.read_csv("logs/run-cells_Fedora-tag-val_loss.csv", delimiter=",", usecols=["Value"])
    
    loss_train = pd.read_csv("logs/run-cells_Fedora-tag-mrcnn_bbox_loss.csv", delimiter=",", usecols=["Value"])
    loss_val = pd.read_csv("logs/run-cells_Fedora-tag-val_mrcnn_bbox_loss.csv", delimiter=",", usecols=["Value"])
    
    
    epochs = range(1,len(loss_train)+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Mask R-CNN BBox Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0,1))
    plt.legend()
    plt.savefig('mrcnn_bbox_loss.png', bbox_inches='tight')
    plt.show()
    
    
    

main_training()
#main_compare()
#main_use()
#main_plot_curves()

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




        # Compute AP
        AP, precision, recall, overlap =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             other_bbox, other_class_id, a, other_mask) # np.ones((1,len(other_class_id)))
      
        APs.append(AP)
        precisions.append(np.mean(precision))
        recalls.append(np.mean(recall))
        overlaps.append(np.mean(overlap))

        #visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
        #visualize.display_instances(image, other_bbox, other_mask, other_class_id, dataset_train.class_names, figsize=(8, 8))
        

"""