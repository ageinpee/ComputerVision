# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:42:37 2017

@author: wilms
"""

import keras
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD


(X_train, y_train), (X_test, y_test) = cifar10.load_data() #CIFAR-10 Datensatz laden

X_train = X_train.astype(np.float32)/255

#DATA AUGMENTATION (imgs)
X_mirrored = np.flip(X_train, 2) #axis should be correct
X_train = np.concatenate([X_train, X_mirrored], axis=0)

#y_train = y_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255
#y_test = y_test.astype(np.float32)/255

y_train = keras.utils.to_categorical(y_train, num_classes=10)

#DATA AUGMENTATION (labels)
y_train = np.concatenate([y_train, y_train], axis=0)

y_val = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', 
                 input_shape=(X_train.shape[1:]), name='conv1'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same', 
                 input_shape=(X_train.shape[1:]), name='conv2'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same', 
                 input_shape=(X_train.shape[1:]), name='conv21'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same', 
                 input_shape=(X_train.shape[1:]), name='conv22'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(256, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy', 
                  optimizer=SGD(lr=0.001, momentum=0.9), 
                  metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=20,
          validation_split = 0.2, verbose=1, 
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0, patience=3), 
                    keras.callbacks.ModelCheckpoint(filepath='./W7.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, 
                                                    save_weights_only=False, 
                                                    mode='auto', period=1)])

model.load_weights('./W7.h5', by_name=True)

score = model.evaluate(X_test, y_val, verbose=1)

print('lr=0.0001')
print(score)