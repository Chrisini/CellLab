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
    plt.savefig(f"C://Users/Prinzessin/Documents/LifeSci/disseration_images/normalized/python/bhist_{mode}_cell138.png", bbox_inches = 'tight')
    image.save(f"C://Users/Prinzessin/Documents/LifeSci/disseration_images/normalized/python/bimg_{mode}_cell138.jpg")
    
    
def normalisation_and_equalisation():
    
    image_original = Image.open("C://Users/Prinzessin/Documents/LifeSci/named_images/cell138.jpg")
    
    # (v1) minmax
    image_min = np.min(image_original)
    image_max = np.max(image_original)
    
    min_scale = 0.0
    max_scale = 1.0
        
    image_scaled_minmax = (image_original - image_min) / (image_max - image_min) * (max_scale - min_scale) + min_scale
    # to pillow
    image_scaled_minmax = (image_scaled_minmax*255.0).astype(np.uint8)
    image_scaled_minmax = Image.fromarray(image_scaled_minmax, 'RGB')
    
    # (v2) pillow autocontrast
    image_scaled_autocont =  ImageOps.autocontrast(image_original, cutoff=0, ignore=None)
    
    # (v3) pillow equalize
    image_scaled_equal =  ImageOps.equalize(image_original)
    
    # save histograms and image results
    save_hist_and_resul(image_original, "original")
    save_hist_and_resul(image_scaled_minmax, "scaled_minmax")
    save_hist_and_resul(image_scaled_autocont, "scaled_autocon")
    save_hist_and_resul(image_scaled_equal, "scaled_equal")

normalisation_and_equalisation()
