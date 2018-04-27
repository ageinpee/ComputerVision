# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:24:44 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)
"""

import numpy as np
import matplotlib as plt
from skimage.io import imread
import glob
from skimage.filters import threshold_otsu
import itertools

val_imgs = glob.glob('./haribo1/hariboVal/*.png')
tr_imgs = glob.glob('./haribo1/hariboTrain/*.png')

val_labels = [x.split('/')[-1].split('.')[0].split('_')[0].split('\\')[1] for x in val_imgs]
tr_labels = [x.split('/')[-1].split('.')[0].split('_')[0].split('\\')[1] for x in tr_imgs]


# general helper functions
def load_imgs():
    for i,img in enumerate(val_imgs):
        val_imgs[i] = imread(img)
    for i,img in enumerate(tr_imgs):
        tr_imgs[i] = imread(img)


def create_seperator():
    sep = ''
    for i in range(15):
        sep += '---------'
    return sep


def compare_correct(labels):
    count = 0
    for i,label in enumerate(labels):
        if val_labels[i] == label:
            count += 1
    return count


def to_binary_hsv(img, lower, upper):
    return (lower < img) & (img < upper)


def to_binary_rgb(img, value):
    return img < value


def show_imgs_hsv(imgs, subplot_x, subplot_y):
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    for i in range(subplot_x):
        for j in range(subplot_y):
            if(imgs[i] is None):
                break
            else:
                ax[i, j].imshow(plt.colors.hsv_to_rgb(imgs[i+j]), 'Greys_r')
    plt.pyplot.show(block=True)


def show_imgs_rgb(imgs, subplot_x, subplot_y):
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    for i in range(subplot_x):
        for j in range(subplot_y):
            if(imgs[i] is None):
                break
            else:
                ax[i, j].imshow(imgs[i+j], 'Greys_r')
    plt.pyplot.show(block=True)


# task functions
def classify_means():
    val_means = np.mean(np.array(val_imgs), axis=(1,2))
    tr_means = np.mean(np.array(tr_imgs), axis=(1,2))
    return val_means, tr_means
    

def compare_means():
    means = classify_means()
    val_means = means[0]
    tr_means = means[1]
    labels = []
    
    for count in range(len(val_means)):
        dists = []
        for i in range(len(tr_means)):
            dists.append(np.linalg.norm(tr_means[i] - val_means[count]))
        labels.append(tr_labels[np.argmin(dists)])
    return labels
    
    
def classify_3dhists():
    val_hists = []
    tr_hists = []
    for img in val_imgs:
        img = img.reshape((img.shape[0]*img.shape[1],3))
        val_hists.append(np.histogramdd(img, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0])
    for img in tr_imgs:
        img = img.reshape((img.shape[0]*img.shape[1],3))
        tr_hists.append(np.histogramdd(img, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0])
    return val_hists, tr_hists


def compare_3dhists():
    hists = classify_3dhists()
    val_hists = hists[0]
    tr_hists = hists[1]
    labels = []
    
    for count in range(len(val_hists)):
        dists = []
        for i in range(len(tr_hists)):
            dists.append(np.linalg.norm(tr_hists[i] - val_hists[count]))
        labels.append(tr_labels[np.argmin(dists)])

    return labels


def convert_to_hsv():
    val_imgs_hsv = map(plt.colors.rgb_to_hsv, val_imgs)
    tr_imgs_hsv = map(plt.colors.rgb_to_hsv, tr_imgs)
    return val_imgs_hsv, tr_imgs_hsv


def binarize():
    imgs_hsv = convert_to_hsv()
    val_imgs_hsv = imgs_hsv[0]
    tr_imgs_hsv = imgs_hsv[1]

    val_binary = list(itertools.starmap(to_binary_hsv, [(img, 30, 140) for img in val_imgs_hsv]))
    tr_binary = list(itertools.starmap(to_binary_hsv, [(img, 30, 140) for img in tr_imgs_hsv]))
    val_binary = [255*img for img in val_binary]
    tr_binary = [255*img for img in tr_binary]

    return val_binary, tr_binary


def binarize_otsu():
    val_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in val_imgs]))
    tr_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in tr_imgs]))
    val_otsu = map(threshold_otsu, val_imgs_greys)
    tr_otsu = map(threshold_otsu, tr_imgs_greys)

    val_binary_otsu = list(itertools.starmap(to_binary_rgb,
                                             [(img, val_otsu[i]) for i,img in enumerate(val_imgs_greys)]))
    tr_binary_otsu = list(itertools.starmap(to_binary_rgb,
                                            [(img, tr_otsu[i]) for i,img in enumerate(tr_imgs_greys)]))
    val_binary_otsu = [255*img for img in val_binary_otsu]
    tr_binary_otsu = [255*img for img in tr_binary_otsu]

    return val_binary_otsu, tr_binary_otsu


def create_bounding_boxes():
    return 'bounding boxes still have to be created'



if __name__ == '__main__':
    load_imgs()
    #print val_labels
    #print compare_means()
    #print compare_correct(compare_means()), 'Labels were chosen correct'
    #print compare_3dhists()
    #print compare_correct(compare_3dhists()), 'Labels were chosen correct'
    #print create_seperator()
    #show_imgs_hsv(binarize()[1], 3, 4)
    create_bounding_boxes()
