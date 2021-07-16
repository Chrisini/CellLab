# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:52:44 2020

@author: Prinzessin
"""

# I guess this is important for neural networks

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def save_hist_and_resul(image, mode):
    
    channels = np.array(image).T
    
    colours = ('r','g','b')
    plt.figure()
    for i, colour in enumerate(colours):      
        # get histogram for each channel
        histo = np.histogram(channels[i].flatten(), bins=256, range=[0, 256])
        x_axis = np.arange(len(histo[0]))
        y_axis = histo[0]
        # bar plot
        plt.bar(x_axis, y_axis, align='center', alpha=0.5, color=colour)

    # save histogram and image results
    #plt.savefig(f"C://Users/Prinzessin/Documents/LifeSci/disseration_images/normalized/python/bhist_{mode}_cell138.png", bbox_inches = 'tight')
    #image.save(f"C://Users/Prinzessin/Documents/LifeSci/disseration_images/normalized/python/bimg_{mode}_cell138.jpg")
    
    
def normalisation_and_equalisation():
    
    image_original = Image.open("C://Users/Prinzessin/Documents/LifeSci/named_images/cell138.jpg")
    
    #image_original = Image.open("C://Users/Prinzessin/Downloads/eye_michi.jpg")
    
    
    # (v1) minmax

    
    image_min = np.min(np.array(image_original)[0])
    image_max = np.max(np.array(image_original)[0])
    
    min_scale = 0.0
    max_scale = 1.0
        
    image_scaled_minmax = (image_original - image_min) / (image_max - image_min) * (max_scale - min_scale) + min_scale
    
    
    image_unscaled_minmax =  image_scaled_minmax * 0.3 + 0.3
    
    
    image_unscaled_minmax = (image_unscaled_minmax*255.0).astype(np.uint8)
    image_unscaled_minmax = Image.fromarray(image_unscaled_minmax, 'RGB')
    
    # to pillow
    image_scaled_minmax = (image_scaled_minmax*255.0).astype(np.uint8)
    image_scaled_minmax = Image.fromarray(image_scaled_minmax, 'RGB')
    
    # (v2) pillow autocontrast
    image_scaled_autocont =  ImageOps.autocontrast(image_original, cutoff=2, ignore=None)
    
    # (v3) pillow equalize
    image_scaled_equal =  ImageOps.equalize(image_original)
    
    # save histograms and image results
    
    save_hist_and_resul(image_unscaled_minmax, "unscaled")
    image_unscaled_minmax.show()
    
    save_hist_and_resul(image_original, "original")
    image_original.show()
    
    save_hist_and_resul(image_scaled_minmax, "scaled_minmax")
    image_scaled_minmax.show()
    save_hist_and_resul(image_scaled_autocont, "scaled_autocon")
    image_scaled_autocont.show()
    #save_hist_and_resul(image_scaled_equal, "scaled_equal")

normalisation_and_equalisation()




"""
def cv_show_histograms(image):
    
    # convert pillow to opencv image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    colours = ('b','g','r')
    plt.figure()
    for i, colour in enumerate(colours):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.plot(hist, color = colour)
        plt.xlim([0,256])
    plt.show()
    
def np_show_histograms_old(image):
    # get 2D arrays of RGB channels
    channels = np.array(image).T
    
    colours = ('r','g','b')
    plt.figure()
    for i, colour in enumerate(colours):      
        
        # make channel 1D by flatten() and get histogram
        hist = np.histogram(channels[i].flatten(), bins=256, range=[0, 256])
        
        plt.plot(hist[1][:256], hist[0], color=colour)
        plt.xlim([0,256])
        
    plt.show()
"""
