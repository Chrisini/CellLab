# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:08:13 2021

@author: Prinzessin
"""

from PIL import Image, ImageChops
import numpy as np

import glob

import random

cell_list = []



def filename_rand(celllist):
    for i in range (0, 50):
        print(random.choice(celllist))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    try:
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
    except Exception as e:
        pass
        err = 9999
        
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err

for filename in glob.glob('C://Users/Prinzessin/Documents/LifeSci/named_images_split/*.jpg'): 
    image_one=Image.open(filename).convert('RGB')
    cell_list.append(filename)
    for filename2 in glob.glob('C://Users/Prinzessin/Documents/LifeSci/original_data/zytotox_08_06_2020_zinksulfat-chrisy/*.jpg'): #assuming gif
        image_two=Image.open(filename2).convert('RGB')
     
        diff = ImageChops.difference(image_one, image_two)
        error_val = mse(np.array(image_one), np.array(image_two))

        if error_val < 100:
            print("images are the same", filename)
            print("images are the same", filename2)
            print("******")


filename_rand(cell_list)