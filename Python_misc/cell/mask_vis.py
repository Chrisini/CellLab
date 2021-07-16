# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:09:51 2020

@author: Prinzessin
"""

from PIL import Image

mask = Image.open('C:/Users/Prinzessin/Downloads/PennFudanPed/PennFudanPed/PedMasks/FudanPed00001_mask.png')
# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's adda color palette to the mask.


pixels = list(mask.getdata())
width, height = mask.size
pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

print(pixels)


mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 0, 0, # index 1 is red
    # 255, 255, 0, # index 2 is yellow
    # 255, 153, 0, # index 3 is orange
])




mask.show()