from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2

# plot histogram with matplotlib + numpy, recommended
def np_show_histograms(image):
    
    channels = np.array(image).T
    
    colours = ('r','g','b')
    plt.figure()
    for i, colour in enumerate(colours):      
        
        histo = np.histogram(channels[i].flatten(), bins=256, range=[0, 256])
            
        y_pos = np.arange(len(histo[0]))
        performance = histo[0]
        
        plt.bar(y_pos, performance, align='center', alpha=0.5, color=colour)

# plot histogram with matplotlib + opencv, not recommended
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

# plot histogram with matplotlib + numpy, not recommended
def np_show_histograms_old(image):
    # get 2D arrays of RGB channels
    channels = np.array(image).T
    
    colours = ('r','g','b')
    plt.figure()
    for i, colour in enumerate(colours):        
        # make channel 1D by flatten() and get histogram
        hist = np.histogram(channels[i].flatten(),   bins=256, range=[0, 256])
        plt.plot(hist[1][:256], hist[0], color=colour)
        plt.xlim([0,256])
    plt.show()

# normalise image with numpy
def numpy_normalisation():
    
    image_original = Image.open("C://Users/Prinzessin/Documents/LifeSci/named_images/cell1.jpg")
    
    image_min = np.min(image_original)
    image_max = np.max(image_original)
    
    min_scale = 0.0
    max_scale = 1.0
        
    image_scaled = (image_original - image_min) / (image_max - image_min) * (max_scale - min_scale) + min_scale
    
    # to pillow
    image_scaled = (image_scaled*255.0).astype(np.uint8)
    image_scaled = Image.fromarray(image_scaled, 'RGB')
    image_scaled.show()

    np_show_histograms(image_original)
    np_show_histograms(image_scaled)
