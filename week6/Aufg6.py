# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:03:28 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import glob
from skimage.filters import gaussian
from skimage.measure import regionprops, label
from skimage.morphology import binary_opening, disk, square
from skimage.transform import resize
from skimage.feature import match_template
from scipy.ndimage.filters import convolve
from scipy.ndimage import sobel
from scipy import misc
import time
from skimage.filters.rank import gradient
from multiprocessing import Process
import bilderGenerator as bG

np.random.seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD


def generate_images():
    training = bG.zieheBilder(500)
    validation = bG.zieheBilder(50)
    return training, validation


def plot_imgs(imgs):
    plt.close('all')
    
    plus = np.where(imgs[0][2] == 1, imgs[0], 0)
    minus = np.where(imgs[0][2] == -1, imgs[0], 0)
    fig, ax = plt.subplots(1,1)
    
    ax.plot(plus[0], plus[1], 'rx')
    ax.plot(minus[0], minus[1], 'bx')
    
    
def linear_class(imgs, w1=0.0001, w2=(-0.0002), b=0.001):
    estim = []  #list of all answer estimations.
    for i in range(len(imgs[0])):
        estim.append(np.sign(imgs[2][i]) == np.sign(w1 * imgs[0][i] + w2 * imgs[1][i] + b))

    return np.sum(estim)/float(len(estim))


def train_neuron(imgs, w1=0.0001, w2=-0.0002, b=0.0001, epochs=1):
    alpha = 0.0000005
    for epoch in range(epochs):
        for i in range(len(imgs[0])):
            x1 = imgs[0][i]
            x2 = imgs[1][i]
            lab = imgs[2][i]
            if not (np.sign(lab) == np.sign(w1 * x1 + w2 * x2 + b)):
                t = lab
                w1 = w1 - alpha * (2 * (w1 * x1 + w2 * x2 + b - t) * x1)
                w2 = w2 - alpha * (2 * (w1 * x1 + w2 * x2 + b - t) * x2)
                b = b - alpha * (2 * (w1 * x1 + w2 * x2 + b - t))
    return w1, w2, b


def plot_border(imgs, twx):     # twx = trained weights X
    plt.close('all')

    plus = np.where(imgs[0][2] == 1, imgs[0], 0)
    minus = np.where(imgs[0][2] == -1, imgs[0], 0)
    fig, ax = plt.subplots(1,1)

    ax.plot(plus[0], plus[1], 'rx')
    ax.plot(minus[0], minus[1], 'bx')

    for x in range(2550):
        for y in range(1280):
            value = (twx[0] * x + twx[1] * y + twx[2])
            if (-0.0001) < value < 0.0001:
                ax.plot(x*0.1, y*0.1, 'g.')
    '''
    Anmerkung der Redaktion:
    WTF!!!! HIER WIRD DER GESAMTE SUCHRAUM ABGEGANGEN! DAS IST EIN GRAUß!!!
    Anmerkung Ende.
    '''

def get_merkmale(imgs, merkmale):
    
    for i in range(imgs.shape[0]):
        merkmale[i][0] = np.mean(imgs[i], axis=(0,1))[0]
        merkmale[i][1] = np.mean(imgs[i], axis=(0,1))[1]
        merkmale[i][2] = np.mean(imgs[i], axis=(0,1))[2]
        
        merkmale[i][3] = np.std(imgs[i], axis=(0,1))[0]
        merkmale[i][4] = np.std(imgs[i], axis=(0,1))[1]
        merkmale[i][5] = np.std(imgs[i], axis=(0,1))[2]
    
    merkmale = merkmale.astype(np.float32)
        

def transform_labels(labels):
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            labels[i] = 0
        if labels[i] == 4:
            labels[i] = 1
        if labels[i] == 8:
            labels[i] = 2

def neural_network():
    d = np.load('./trainingsDatenFarbe2.npz')
    tr_imgs = d['data']
    tr_labels = d['labels']

    d = np.load('./validierungsDatenFarbe2.npz')
    val_imgs = d['data']
    val_labels = d['labels']
    
    tr_merkmale = np.zeros((60,6))
    val_merkmale = np.zeros((30,6))
    
    get_merkmale(tr_imgs, tr_merkmale)
    get_merkmale(val_imgs, val_merkmale)
    
    transform_labels(val_labels)
    transform_labels(tr_labels)
    
    Y_train = np_utils.to_categorical(tr_labels, 3)
    Y_test = np_utils.to_categorical(val_labels, 3)
    
    model = Sequential()
    model.add(Dense(8, activation = 'relu', name = 'fc1', input_shape=(6,)))
    model.add(Dense(8, activation = 'relu', name = 'fc2'))
    model.add(Dense(3, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=SGD(lr=0.000005, momentum=0.9), 
                  metrics=['accuracy'])
    
    model.fit(tr_merkmale, Y_train, batch_size=1, nb_epoch=500, verbose=1)
    score = model.evaluate(val_merkmale, Y_test, verbose=1)
    print(score)
    
    '''
    Die Ergebnisse sind mit 0.63 accuracy etwas besser als die 0.5?
    mit dem NN-Klassifikator
    '''


if __name__ == '__main__':
    """
    images = generate_images()
    tr, val = generate_images()
    plot_imgs(images)
    plt.show(block=True)
    print 'linear classification of training images: ', linear_class(tr)
    print 'linear classification of validation images: ', linear_class(val)
    tw = train_neuron(tr)   # tw = trained weights
    print 'linear classification of validation images with trained weights: ', \
        linear_class(val, tw[0], tw[1], tw[2])
    tw2 = train_neuron(tr, np.random.normal(0, 0.001), np.random.normal(0, 0.001), 0)
    print 'linear classification of validation images with trained weights but modified starting values: ', \
        linear_class(val, tw2[0], tw2[1], tw2[2])
    tw3 = train_neuron(tr, np.random.normal(0, 0.001), np.random.normal(0, 0.001), 0, 100)
    print 'linear classification of validation images with trained weights over 100 epochs: ', \
        linear_class(val, tw3[0], tw3[1], tw3[2])
    plot_border(images, tw3)
    plt.show(block=True)
    """
    neural_network()