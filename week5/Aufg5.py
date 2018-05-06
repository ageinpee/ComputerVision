# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:05:44 2018

@author: 6peters
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


def bad_convolve(img):
    out = np.empty(img.shape)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            neighbors = []            
            for i in range(-1, 2):
                for j in range(-1,2):
                    if x+i in range(img.shape[0]) and y+j in range(img.shape[1]):
                        neighbors.append(img[x+i][y+j])
                    else: neighbors.append(0)
            out[x][y] = np.mean(neighbors)

    return out


def scipy_convolve(img, size_x, size_y):
    mask = np.ones((size_x, size_y))*(1/float(size_x*size_y))
    out = convolve(img, mask)
    
    return out
    
    
def sobeling(img):
    sobel_x = sobel(img, axis = -1)
    sobel_y = sobel(img, axis = 0)
    
    return sobel_x, sobel_y
    
    
def create_gradients_img(img_x, img_y):
    out = np.hypot(img_x, img_y)
    
    return out


def bad_template_matching_map(img, template):
    out = np.empty(img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            similarities = []
            for i in range(int((0-template.shape[0])/2.0), int((template.shape[0])/2.0)+1):
                for j in range(int((0-template.shape[1])/2.0), int((template.shape[1])/2.0)+1):
                    if x + i in range(img.shape[0]) and y + j in range(img.shape[1]):
                        similarities.append(int(img[x+i][y+j]-int(template[i][j])))
                        print x, y
                    else:
                        similarities.append(0)
                        print 'not inside'
            out[x][y] = np.mean(similarities)
    return out


def template_matching(img, template):
    return match_template(img, template)


def find_max_template(matched_img):
    return np.unravel_index(np.argmax(matched_img), matched_img.shape)


def wheres_wally():
    wally_matching_map = template_matching(wheres_w, wally)
    wally_match = find_max_template(wally_matching_map)
    fig1, ax1 = plt.subplots(1, 1)
    ax1.imshow(wheres_w)
    ax1.plot(wally_match[1], wally_match[0], color='red', marker='x', linestyle='dashed', linewidth=2, markersize=12)


if __name__ == '__main__':
    lenna_noisy = imread('./noisyLenna.png')
    lenna = imread('./Lenna.png')
    templ_auge = imread('./auge.png')
    wheres_w = imread('./whereIsWally1.jpg')
    wally = imread('./wally.png')

    #start = time.time()
    #imshow(bad_convolve(lenna_noisy))
    #bc = time.time()
    #bc_diff = bc-start
    #print bc_diff

    fig, ax = plt.subplots(5, 2)

    start = time.time()
    ax[0, 0].imshow(scipy_convolve(lenna_noisy, 30, 30), 'Greys_r')
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

    templ_match = template_matching(lenna, templ_auge)
    ax[4, 0].imshow(templ_match)
    ax[4, 0].set_title('implemented template matching', size=7)
    max_match = find_max_template(template_matching(lenna, templ_auge))
    ax[4, 1].imshow(templ_match)
    ax[4, 1].plot(max_match[1], max_match[0], 'rx')
    ax[4, 1].set_title('maximal template match coordinates', size=7)
    '''
        Aufg. 5.2: Gefunden wird nur das rechte Auge. Gefunden ist auch relativ. 
        Genau genommen wird nur der Pixel gefunden, bei der die geringste Differnz der
        Bildwerte vorliegt.
    '''
    print 'Die Koordinaten fuer den maximalen Match-Wert lauten: ', max_match
    wheres_wally()

    plt.show(block=True)
