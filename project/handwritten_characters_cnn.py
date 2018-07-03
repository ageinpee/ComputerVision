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
    print(X_train[0].shape)
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                  input_shape=X_train.shape[1:], name='conv1'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                  name='conv2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  name='conv3'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  name='conv4'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dense(26, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])
    return model

    """
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', padding='same',
                     input_shape=(28, 28, 1), name='conv1'))
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', padding='same',
                     input_shape=(28, 28, 1), name='conv2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64, (2,2), activation='relu', padding='same',
                     input_shape=(28, 28, 1), name='conv21'))
    model.add(keras.layers.Conv2D(64, (2,2), activation='relu', padding='same',
                     input_shape=(28, 28, 1), name='conv22'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dense(26, activation='softmax', name='output'))
    
    model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                      metrics=['accuracy'])
    
    return model
    """

def train_cnn(model, X_train, Y_train):
    '''
    model.fit(X_train, Y_train, batch_size=32, epochs=200,
          validation_split = 0.2, verbose=1, 
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, patience=5),
                    keras.callbacks.ModelCheckpoint(filepath='./project.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, 
                                                    save_weights_only=False, 
                                                    mode='auto', period=1)])
    '''
    model.fit(X_train, Y_train, batch_size=32, epochs=20,
          validation_split = 0.2, verbose=1)
"""
Main functions:
--------------------------------------------------------------------------
"""

if __name__ == '__main__':
    images = image_ops.load_images_npz(input("Enter a file path for the data: "))
    #images = np.load('C:/Users/Moritz Lahann/Desktop/STUDIUM/PRAKTIKUM COMPUTERVISION/DATA/TEST.npz')
    '''
    for key in images:
        images[key] = np.reshape(images[key][:,:,:,0], newshape=(1000,28,28,1))
        images[key] = to_binary(images[key], 127)*255

    X_train = np.zeros(shape=(20800, 28, 28, 1))
    Y_train = []
    
    X_test = np.zeros(shape=(5200, 28, 28, 1))
    Y_test = []
    for i, key in enumerate(images):
        for j in range(0, 800):
            rnd_index = random.choice(range(0, 800-j))
            X_train[i*800+j] = images[key][rnd_index]
            images[key] = np.delete(images[key], rnd_index)
            Y_train.append(key)
        for j in range(0, 200):
            rnd_index = random.choice(range(0, 200-j))
            X_test[i*200+j] = images[key][rnd_index]
            images[key] = np.delete(images[key], rnd_index)
            Y_test.append(key)

    print(X_train.shape)
    print(X_test.shape)
    '''
    
    keylen = images["A"].shape[0]
    print(keylen)
    
    X_train = np.zeros(shape=(26*keylen, 28, 28, 4))
    Y_train = []
    
    for i, key in enumerate(images):
        for j in range(keylen):
            X_train[i*keylen+j] = images[key][j]
            Y_train.append(key)
    
    #print(X_train)
    for i in range(len(Y_train)):
        Y_train[i] = ord(Y_train[i])-65
    print(Y_train)
    '''
    for i in range(len(Y_test)):
        Y_test[i] = ord(Y_test[i])-65
    print(Y_test)
    '''
    #convert to suitable for CNN
    Y_train = keras.utils.to_categorical(Y_train, 26)
    #Y_test = keras.utils.to_categorical(Y_test, 26)

    X_train = X_train.astype(np.float32)/255
    #X_test = X_test.astype(np.float32)/255


    #only run for new parameters
    model = compile_cnn(X_train)
    train_cnn(model, X_train, Y_train)
    
    #model.load_weights('./project.h5', by_name=True)

    '''
    score = model.evaluate(X_test, Y_test, verbose=1)
    
    print(score[1])

    img_print = []
    label_print = []
    for i in range(26):
        img = X_train[i*800]
        label = Y_train[i*800]
        print(img.shape)
        img = img[:,:,0]
        print(img.shape)
        img_print.append(img)
        label_print.append(label)
    image_ops.show_images(img_print, 6, 6)
    print(label_print)
    #plt.imshow(X_train[4000], "Greys_r")  # test
    plt.show()
    '''
