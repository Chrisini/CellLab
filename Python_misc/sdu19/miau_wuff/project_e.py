#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:20:32 2019

@authors:Emilie Winkler, Christina Bornberg, Dimitri Franz, Isabel Vinterbladh
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import math as math
import numpy as np
#%% Task 1 
# After the data had been divided up in different directories for 
#training (1000 cats, 1000 dogs), validation (250 cats, 250 dogs) 
#and testing (250 cats, 250 dogs) we all created our neural networks to try out 
#different things and then got together to compare and see what worked the best.

#%%Task 3: Emilie Winkler
# Convolutional layers were used since they are good when working with images. 
# Through testing was it concluded that 2-3 convolutional layers where the best,
# in the end we went with 3. Maxpooling was most succesfull when implemented 
#between each convolutional layer. For activation function best results were 
#recieved with relu and sigmoid in the output, sigmoid is good for binary classification.
#However also tanh was tested but did not give as good results.
# After the flatten a Dense layer was put and given a dropout, different values for 
# the dropout was tested but 0.2 worked the best. 
# Lastly we also tried to add regularizators, both L1 and L2 seperatly, however
#that only resulted in worse accuracy and higher loss so we decided to not use it.

visible = Input(shape=(150, 150, 1))

conv1 = Conv2D(32, kernel_size=4, activation='relu', name= 'Conv1')(visible)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, kernel_size=4, activation='relu', name= 'Conv2')(pool1)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(32, kernel_size=4, activation='relu', name= 'Conv3')(pool2)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

flat = Flatten()(pool3)

hidden1 = Dense(64, activation='relu', name= 'hidden')(flat)

hidden2 = Dropout(0.2)(hidden1)

output = Dense(1, activation='sigmoid', name='output')(hidden2)

model = Model(inputs=visible, outputs=output)

# Christina Bornberg
# summarize of model
print(model.summary())

# For optimizer were rmsprop and adam both tested. Adam should work good for binary
# cases and we also concluded that it gave better result than rmsprop.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%% Task 2 
# Different batch_sizes were tried, a bigger one resulted in less computational time
# but a smaller one as 16 gave better result. 
batch_size = 16

# The datagen for the training and validation directories were created
train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

# All images were resized to 150x150 and the color_mode grayscale gave the better 
#result for us. Which we assume could be that it should not matter what color the
# cats and dogs are, it should still be able to categorize it and the grayscale 
# has less information than a full color picture so it decreases the computational
#time.
train_gen = train_data.flow_from_directory(
        'train',  
        target_size=(150, 150),
		color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')  

validation_gen = test_data.flow_from_directory(
        'validation',
        target_size=(150, 150),
		color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')

#Isabel Vinterbladh
# Running the model and saving it
epochsteps= math.ceil(2000/batch_size)
valsteps= math.ceil(500/ batch_size)

his= model.fit_generator(
        train_gen,
        steps_per_epoch= epochsteps,
        epochs=40,
        validation_data=validation_gen,
        validation_steps= valsteps)

# The best results that we got was an training accuracy of 0.95 and validation accuracy 
# of 0.79. So this would give an estimation of 20 % for the generalization error. 
model.save('model.h5')

#%%Task 4: 
# Plotting the accuracy and loss
history_dict = his.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.plot(epochs, acc_values, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'r', label='Validation accuracy')

plt.title('Training and validation loss and accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()

#%%TEST MODEL
# Dimitri Franz
test_generator = test_data.flow_from_directory(
    directory="./dataset_catdog_split/dataset_catdog/test",
    target_size=(150, 150),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

test_generator_dog = test_data.flow_from_directory(
    directory="./dataset_catdog_split/dataset_catdog/test_dogs",
    target_size=(150, 150),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

test_generator_cat = test_data.flow_from_directory(
    directory="./dataset_catdog_split/dataset_catdog/test_cats",
    target_size=(150, 150),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

STEP_SIZE_TEST= 500 // test_generator.batch_size
test_generator.reset()
test_generator_dog.reset()
test_generator_cat.reset()
pred=model.predict_generator(test_generator,
                             steps=STEP_SIZE_TEST,
                             verbose=1)
pred_cat=model.predict_generator(test_generator_cat,
                             steps=STEP_SIZE_TEST,
                             verbose=1)
pred_dog=model.predict_generator(test_generator_dog,
                             steps=STEP_SIZE_TEST,
                             verbose=1)
#%% PLOT TEST RESULTS
# 0 and 1 correspond to cat and dog respectively

fig1 = plt.figure()
ax1 = fig1.gca()
n, x, _ = ax1.hist(pred, bins=20, color='k', alpha=0.8)

n_dog, x_dog = np.histogram(pred_dog, x)  
n_cat, x_cat = np.histogram(pred_cat, x)

ax1.plot([(x_dog[idx]+x_dog[idx+1])/2 for idx in range(20)], n_dog, color='b', label='Dogs only')
ax1.plot([(x_cat[idx]+x_cat[idx+1])/2 for idx in range(20)], n_cat, color='r', label='Cats only')

plt.xticks([0, 1], ['Cat', 'Dog'])
plt.ylabel('Counts')
plt.ylim([0, 275])
plt.legend(frameon= False)
plt.tight_layout()

#Conclusion: After using our model on the test data we can see that the network
# thinks there are more pictures with dogs on than the 250 ones, this can be seen
# in our visualization graph. Therefore, the model predicts that there are less
# than 250 pictures of cats. However, the results are still good and resonable 
#considering our generalization error.   