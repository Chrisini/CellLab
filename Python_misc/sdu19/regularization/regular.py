import keras
from keras import layers
from keras import optimizers
from keras import regularizers  
#from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

print(train_labels)

# one hot encoding
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

def create_model():
        
    model = keras.Sequential()
    model.add(layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,))) #input
    model.add(layers.Dense(16, activation="relu")) #hidden layer
    model.add(layers.Dense(1, activation="sigmoid")) #output layer
    
    #What accuracy do you get? 
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
    
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
    
    #Save your model
    model.save('my_model.h5') # creates a HDF5 file 'my_model.h5'del model 
    del model


def plot_stuff(epochs, line1, line2, line3):
    
    epochs = range(1, len(line1) + 1)
    plt.plot(epochs, line1, 'bo', label='Training acc')
    plt.plot(epochs, line2, 'b', label='Validation acc')
    plt.plot(epochs, line3, 'r', label='Regularizer')
    plt.title('Chart')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    
######################################## PART 2

#Load your model again, and use model.evaluate to get the accuracy
def load_h5_model(X, Y):
    
    loaded_model = load_model('my_model.h5')
    print(loaded_model.summary())
    loaded_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    loaded_model.load_weights('my_model.h5')
    print(loaded_model.metrics_names) # output: loss, acc
    score = loaded_model.evaluate(X, Y, batch_size=32, verbose=0)
    print(score)
    
    #Try adding some regularization, can you increase the accuracy of the model?
    #loaded_model.add(layers.Dense(16, input_dim=64,
     #           kernel_regularizer=regularizers.l2(0.01),
      #          activity_regularizer=regularizers.l1(0.01)))
    
    loaded_model.get_layer('dense_4').kernel_regularizer = regularizers.l2(0.01) 
    loaded_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    
    history = loaded_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
    
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

    
    # --------- PLOT ----------

#You can try adding dropout between some layers, where do you find it makes a difference?
#How about L2 regularization?
#Vary the number of units, the amount of layers, activation functions, etc. to obtain the best accuracy you can
# create_model();

load_h5_model(test_data, test_labels)