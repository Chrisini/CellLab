import keras
from keras import layers
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

NUM_WORDS = 10000

#The goal of this exercise is to look at movie reviews and tell if the 
#review is positive or negative (Binary classification)
#Create a model with 3 fully connected layers. The input layer must have 
#16 units, the hidden layer must have 16 units, and the output layer must 
#have 1 unit

#You can load the model with the following code
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

print("Train Data:")
#print(train_data)
print()

print("Test Data:")
#print(test_data)
print()

#To use the data with a fully connected network, you will have to one-hot-encode the data.
#The following code does that for you
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

model = keras.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,))) #input
model.add(layers.Dense(16, activation="relu")) #hidden layer
model.add(layers.Dense(1, activation="sigmoid")) #output layer
# relu - hidden
# sigmoid und softmax - for outputs


print("Train Data:")
print(train_data)
print()

print("Test Data:")
print(test_data)

#What accuracy do you get? 
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#print(accuracy)

#epoch = number of points in the diagram (?)
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
#batchsize - not to small, not to big - use higher when inputs variieren
# 32 or 64 is good 2^n

history_dict= history.history
acc_values= history_dict['acc']
val_acc_values= history_dict['val_acc']

epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

#Do you think this is good? 
#What happens if you change the number of units?

#Save your model
model.save('my_model.h5') # creates a HDF5 file 'my_model.h5'del model 
# deletes the existing model
# returns a compiled model
# identical to the previous 
onemodel = load_model('my_model.h5')