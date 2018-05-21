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

def generate_images():
    tr = bG.zieheBilder(500)
    val = bG.zieheBilder(50)
    return tr, val
    

def plot_imgs(imgs):
    plt.close('all')
    
    plus = np.where(imgs[0][2]==1, imgs[0], 0)
    minus = np.where(imgs[0][2]==-1, imgs[0], 0)
    fig, ax = plt.subplots(1,1)
    
    '''
    plus_ind = []
    minus_ind= []  
    for i in range(plus.len):
        plus_ind.append([plus[0][i], plus[1][i]])
    for i in range(minus.len):
        minus_ind.append([minus[0][i], minus[1][i]])
    '''
    
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


if __name__ == '__main__':
    images = generate_images()
    tr, val = generate_images()
    plot_imgs(images)
    plt.show(block=True)
    print 'linear classification of training images: ', linear_class(tr)
    print 'linear classification of validation images: ', linear_class(val)
    tw = train_neuron(tr)   #tw = trained weights
    print 'linear classification of validation images with trained weights: ', \
        linear_class(val, tw[0], tw[1], tw[2])
    tw2 = train_neuron(tr, np.random.normal(0, 0.001), np.random.normal(0, 0.001), 0)
    print 'linear classification of validation images with trained weights but modified starting values: ', \
        linear_class(val, tw2[0], tw2[1], tw2[2])
    tw3 = train_neuron(tr, np.random.normal(0, 0.001), np.random.normal(0, 0.001), 0, 100)
    print 'linear classification of validation images with trained weights over 100 epochs: ', \
        linear_class(val, tw3[0], tw3[1], tw3[2])
