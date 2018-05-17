# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:03:28 2018

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
from multiprocessing import Process
import bilderGenerator as bG

def generate_images():
    tr = bG.zieheBilder(500)
    val = bG.zieheBilder(50)
    return tr, val
    

def plot_imgs():
    imgs = generate_images()
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
    
    
def eval():
    tr, val = generate_images()
    y_tr = []
    y_val = []
    for i, img in enumerate(tr):
        y_tr.append(np.sign(tr[2][i])==np.sign(0.0001*img[0]+(-0.0002)*img[1]+0.001))
    for img in val:
        y_val.append(np.sign(val[2][i])==np.sign(0.0001*img[0]+(-0.0002)*img[1]+0.001))
    
    return np.sum(y_tr)/float(len(y_tr)), np.sum(y_val)/float(len(y_val))


plot_imgs()
print eval()
    