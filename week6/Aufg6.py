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
    
    
def linear_class(tr, val):
    y_tr = []
    y_val = []
    for i, img in enumerate(tr):
        y_tr.append(np.sign(tr[2][i]) == np.sign(0.0001*img[0]+(-0.0002)*img[1]+0.001))
    for j, img in enumerate(val):
        y_val.append(np.sign(val[2][j]) == np.sign(0.0001*img[0]+(-0.0002)*img[1]+0.001))
    
    return np.sum(y_tr)/float(len(y_tr)), np.sum(y_val)/float(len(y_val))


def train_neuron(imgs, w1=0.0001, w2=-0.0002, b=0.0001, epochs=1):
    alpha = 0.0000005
    t = 1
    for epoch in range(epochs):
        for i,img in enumerate(imgs):
            if np.sign(img[2][i]) == np.sign(w1*img[0]+w2*img[1]+b):
                print "correct solution"
            else:
                t = (w1*img[0]+w2*img[1]+b)
                print t
                w1 = w1 - alpha * (2*(w1*img[0]+w2*img[1]+b-t)*img[0])
                w2 = w2 - alpha * (2*(w1*img[0]+w2*img[1]+b-t)*img[1])
                b = b - alpha * (2*(w1*img[0]+w2*img[1]+b-t))


if __name__ == '__main__':
    images = generate_images()
    tr, val = generate_images()
    plot_imgs(images)
    plt.show(block=True)
    print linear_class(tr, val)
    train_neuron(tr)

