# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi(6989075)

This file is supposed to host the script for a convolutional neural network. This cnn
    shall classify handwritten letters.

"""

"""
Imports:
--------------------------------------------------------------------------
"""
# System:
import random

# Scientific:
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import keras

# Local
import image_ops

# Set random seed for reproducible results
np.random.seed(4505918)
from tensorflow import set_random_seed
set_random_seed(4505918)


"""
Helping functions:
--------------------------------------------------------------------------
"""

def compile_cnn(X_train):
    model = keras.models.Sequential()
    print(X_train[0].shape)
    
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  input_shape=X_train.shape[1:], name='conv1'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
        
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  name='conv2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  name='conv3'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                  name='conv4'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                  name='conv5'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu', name='fc1'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.5))
    
    model.add(keras.layers.Dense(26, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model

def train_cnn(model, X_train, Y_train):
    model.fit(X_train, Y_train, batch_size=32, epochs=200,
          validation_split = 0.2, verbose=1,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, patience=3),
                    keras.callbacks.ModelCheckpoint(filepath='./project.h5', 
                                                    monitor='val_loss', 
                                                    verbose=0, 
                                                    save_best_only=True, 
                                                    save_weights_only=False, 
                                                    mode='auto', period=1)])
    
"""
Main functions:
--------------------------------------------------------------------------
"""

if __name__ == '__main__':
    #load up image data
    images = image_ops.load_images_npz(input("Enter file path for .npz files: "))
    
    #determine sample size per class
    keylen = images["A"].shape[0]
    print(keylen)
    
    X = np.zeros(shape=(26*keylen, 28, 28, 4))
    Y = []
    
    #convert dict into X, Y lists, convert character keys into ints (0-25)
    for i, key in enumerate(images):
        for j in range(keylen):
            X[i*keylen+j] = images[key][j]
            Y.append(ord(key)-65)
    
    #randomly assign into test and training set
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=4505918)
    
    #convert to data to suitable for CNN
    Y_train = keras.utils.to_categorical(Y_train, 26)
    Y_test = keras.utils.to_categorical(Y_test, 26)

    X_train = X_train.astype(np.float32)/255
    X_test = X_test.astype(np.float32)/255


    #compile and train model (only run for new parameters)
    model = compile_cnn(X_train)
    train_cnn(model, X_train, Y_train)
    
    #load weights and evaluate model (add confusion matrix here)
    model.load_weights('./project.h5', by_name=True)
    score = model.evaluate(X_test, Y_test, verbose=1)
    
    print("Test dataset accuracy: " + score[1])