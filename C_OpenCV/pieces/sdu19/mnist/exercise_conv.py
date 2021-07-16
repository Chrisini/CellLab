#exercise_conv.py

import numpy as np
from keras import models
from keras import layers
import seaborn as sns
from keras.datasets import mnist

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D



#For this exercise you will use the MNIST dataset
#https://keras.io/datasets/#mnist-database-of-handwritten-digits
#This dataset contains images of handwritten digits 0-9
#Your task is to make a CNN to determine the digits in the images
#You can load the dataset with

(X_train, y_train), (X_test, y_test) = mnist.load_data() # load MNIST data

#Look at the data, do you have to do anything before you feed it to your network?
#You can plot the data with the following commands if you install Seaborn (pip install seaborn)



visible = layers.Input(shape = (64,64,1)) # define input 64x64 - b/w

conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2 ,2))(conv2)

flat = layers.Flatten()(pool2)

hidden = layers.Dense(10, activation='relu')(flat)
output = layers.Dense(1, activation='sigmoid')(hidden)
model = models.Model(inputs=visible, outputs=output) # creating model

print(model.summary()) # summarize layers




sns.heatmap(X_train[1,:,:])


#After you've made your CNN and trained it, try playing around with it. Can you improve the accuracy? 
#Can you visualize the filters? Can you set up Tensorboard, so you can plot the training / validation performance?

