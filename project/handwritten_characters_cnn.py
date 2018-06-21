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

import keras

# Local
import image_ops


"""
Helping functions:
--------------------------------------------------------------------------
"""


def to_binary(img, value):
    return img < value


def compile_cnn(X_train):
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                     input_shape=(X_train.shape[1:]), name='conv1'))
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                     input_shape=(X_train.shape[1:]), name='conv2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                     input_shape=(X_train.shape[1:]), name='conv21'))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                     input_shape=(X_train.shape[1:]), name='conv22'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dense(26, activation='softmax', name='output'))
    
    model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), 
                      metrics=['accuracy'])
    
    return model

def train_cnn(model, X_train, Y_train):
    model.fit(X_train, Y_train, batch_size=32, epochs=20,
          validation_split = 0.2, verbose=1, 
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, patience=3), 
                    keras.callbacks.ModelCheckpoint(filepath='./project.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, 
                                                    save_weights_only=False, 
                                                    mode='auto', period=1)])
"""
Main functions:
--------------------------------------------------------------------------
"""

if __name__ == '__main__':
    images = image_ops.load_images_npz(input("Enter a file path for the data: "))

    for key in images:
        print(len(images[key]))
    X_train = []
    Y_train = []
    
    X_test = []
    Y_test = []
    for i, key in enumerate(images):
        for j in range(0, 800):
            rnd_index = random.choice(range(0, 800-j))
            X_train.append(images[key][rnd_index])
            images[key] = np.delete(images[key], rnd_index)
            Y_train.append(key)
        for j in range(0, 200):
            rnd_index = random.choice(range(0, 200-j))
            X_test.append(images[key][rnd_index])
            images[key] = np.delete(images[key], rnd_index)
            Y_test.append(key)
        
    #binarize images
    for img in X_train:
        to_binary(img, 255)
    for img in X_test:
        to_binary(img, 255)
        
    for i in range(len(Y_train)):
        Y_train[i] = ord(Y_train[i])-65
    print(Y_train)
    for i in range(len(Y_test)):
        Y_test[i] = ord(Y_test[i])-65
    print(Y_test)
    
    #convert to suitable for CNN
    Y_train = keras.utils.to_categorical(Y_train, 26)
    for img in X_train:
        np.reshape(img, (28, 28))
        print(img.shape)
    X_train = np.array(X_train)/255
    Y_test = keras.utils.to_categorical(Y_test, 26)
    X_test = np.array(X_test)/255
    
    #only run for new parameters
    model = compile_cnn(X_train)    
    train_cnn(model, X_train, Y_train)
    
    model.load_weights('./W7.h5', by_name=True)

    score = model.evaluate(X_test, Y_test, verbose=1)    
    
    print(score[1])
    plt.imshow(images["A"][0])  # test
    plt.show()
