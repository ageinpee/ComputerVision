# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:05:44 2018

@author: 6peters
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import glob
from skimage.filters import threshold_otsu, gaussian
import itertools 
from skimage.measure import regionprops, label
from skimage.morphology import binary_opening, disk, square
from skimage.transform import resize
from scipy.ndimage.filters import convolve
from scipy.ndimage import sobel
from scipy import misc
import time
from skimage.filters.rank import gradient


lenna_noisy = imread('./noisyLenna.png')
lenna = imread('./Lenna.png')


def bad_convolve():
    out = np.empty(lenna_noisy.shape)
    
    for x in range(lenna_noisy.shape[0]):
        for y in range(lenna_noisy.shape[1]):
            neighbors = []            
            for i in range(-1, 2):
                for j in range(-1,2):
                    if x+i in range(lenna_noisy.shape[0]) and y+j in range(lenna_noisy.shape[1]):
                        neighbors.append(lenna_noisy[x+i][y+j])
                    else: neighbors.append(0)
            out[x][y] = np.mean(neighbors)
    return out


def scipy_convolve(size_x, size_y):
    mask = np.ones((size_x, size_y))*(1/float(size_x*size_y))
    out = convolve(lenna_noisy, mask)
    
    return out
    
    
def sobeling(img):
    sobel_x = sobel(img, axis = -1)
    sobel_y = sobel(img, axis = 0)
    
    return sobel_x, sobel_y
    
    
def create_gradients_img(img_x, img_y):
    out = np.hypot(img_x, img_y)
    
    return out
    
    
    
if __name__ == '__main__':
    #start = time.time()
    #imshow(bad_convolve())
    #bc = time.time()
    #bc_diff = bc-start
    #print bc_diff

    fig, ax = plt.subplots(4, 2)

    start = time.time()
    ax[0, 0].imshow(scipy_convolve(30, 30), 'Greys_r')
    ax[0, 0].set_title('convolved Lenna.png', size=7)
    sc = time.time()
    sc_diff = sc - start
    print sc_diff

    sobels = sobeling(lenna)
    ax[1, 0].imshow(sobels[0], 'Greys_r')
    ax[1, 0].set_title('x-axis sobel-filtering on Lenna.png', size=7)
    ax[1, 1].imshow(sobels[1], 'Greys_r')
    ax[1, 1].set_title('y-axis sobel-filtering on Lenna.png', size=7)
    
    sobels = sobeling(lenna_noisy)
    ax[2, 0].imshow(sobels[0], 'Greys_r')
    ax[2, 0].set_title('x-axis sobel-filtering on noisyLenna.png', size=7)
    ax[2, 1].imshow(sobels[1], 'Greys_r')
    ax[2, 1].set_title('y-axis sobel-filtering on noisyLenna.png', size=7)
    
    sobels = sobeling(gaussian(lenna_noisy, 5))
    ax[3, 0].imshow(sobels[0], 'Greys_r')
    ax[3, 0].set_title('x-axis sobel+gaussian on noisyLenna.png', size=7)
    ax[3, 1].imshow(sobels[1], 'Greys_r')
    ax[3, 1].set_title('y-axis sobel+gaussian on noisyLenna.png', size=7)

    ax[0, 1].imshow(create_gradients_img(sobels[0], sobels[1]))
    ax[0, 1].set_title('gradients of noisyLenna.png after sobel+gaussian', size=7)

    plt.show(block=True)