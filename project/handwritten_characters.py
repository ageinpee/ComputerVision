# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi(6989075)

This file is supposed to host the script for a classification problem approach.
    The problem is to classify handwritten letters. For solving the problem different
    image augmentations will be used.
"""

"""
Imports:
------------------------------------------------------------------------------------------------------------------------
"""
#System:
import random

#Scientific
import numpy as np
import matplotlib.pyplot as plt

#Local
import image_ops

#Miscellaneous
import time


"""
Helping functions:
------------------------------------------------------------------------------------------------------------------------
"""


def to_binary(img, value):
    return img < value


def histogram_x(img):
    x_hist = []
    for i in img:
        x_hist.append(sum(i))
    return x_hist

def histogram_y(img):
    y_hist = []
    for i in np.transpose(img):
        y_hist.append(sum(i))
    return y_hist

"""
IMPORTANT NOTE: Input is either a list or ndarray. Output is a tuple of 2 lists of images
"""
def create_tr_val_data(imgs, num_tr=800, num_val=200):
    tr = []
    val = []
    for i in range(0, num_tr):
        choice = random.choice(range(len(imgs)))
        tr.append(imgs[choice])
        image_ops.remove_array(imgs, imgs[choice])
    for i in range(0, num_val):
        choice = random.choice(range(len(imgs)))
        val.append(imgs[choice])
        image_ops.remove_array(imgs, imgs[choice])
    return tr, val


"""
Main functions:
------------------------------------------------------------------------------------------------------------------------
"""


"""
takes a list of images as input and creates a image stack as described in ideas.txt
"""
def create_stack(imgs):
    stack = np.zeros((28, 28, 4))#imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[3]))
    for i, img in enumerate(imgs):
        stack = (stack + img)
    stack = stack / len(imgs)
    return stack


def validate_stack(stack, imgs):
    diff_list = []
    for i, img in enumerate(imgs):
        diff_list.append(np.mean(stack - img))
    return diff_list

def projection(tr_imgs, val_imgs, tr_labels):
    tr_x_hists = []
    tr_y_hists = []
    j = 0
    for img in tr_imgs:
        image_ops.print_progress_bar(j, len(tr_imgs) - 1, 
                                     prefix='Processing training images',
                                     suffix='Complete', length=50)
        tr_x_hists.append(histogram_x(img))
        tr_y_hists.append(histogram_y(img))
        j += 1
    
    val_x_hists = []
    val_y_hists = []
    j = 0
    for img in val_imgs:
        image_ops.print_progress_bar(j, len(val_imgs) - 1, 
                                     prefix='Processing validation images',
                                     suffix='Complete', length=50)
        val_x_hists.append(histogram_x(img))
        val_y_hists.append(histogram_y(img))
        j += 1
    
    return validate_projection(np.hstack((tr_x_hists, tr_y_hists)), np.hstack((val_x_hists, val_y_hists)), tr_labels)
    

def validate_projection(tr, val, tr_xy_labels):
    labels = []
    j = 0
    for count in range(len(val)):
        start = time.perf_counter()

        dists = []
        for i in range(len(tr)):
            dists.append(np.linalg.norm(tr[i] - val[count]))
        labels.append(tr_xy_labels[np.argmin(dists)])
        
        end = time.perf_counter()
        #note this is pretty inaccurate, maybe do for every X iterations
        remaining = (end - start) * (len(val) - j)
        image_ops.print_progress_bar(j, len(val) - 1, 
                                     prefix='Assigning labels',
                                     suffix='{0}s remaining'.format(round(remaining)), length=50)
        j += 1
    
    return labels    
"""
Main execution
------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    images = image_ops.load_images_npz(input("Enter a file path for the npz-data: "))
    
    #for key in images:
    #    print(key, len(images[key]))
    #    images[key] = image_ops.augment_images(images[key], 200, key)
    #    image_ops.save_images_npz("data/Data_Test/Data_" + key, images[key])
    
    '''
    train = images
    validate = images
    count = 0
    for key in images:
        image_ops.print_progress_bar(count, 25, prefix='Preparing tr/val-data for {0}'.format(key),
                                     suffix='Complete', length=50)
        images[key] = to_binary(images[key], 127)*255   # binarization of all images.
        train[key], validate[key] = create_tr_val_data(images[key], 180, 20)   # no parameters = standard of 800/200 tr/val
                                                                      # all tr/val lists are still ordered by label
        count += 1

    for key in train:
        print("train", key, len(train[key]))
        print("validate", key, len(validate[key]))
    
    #image_ops.show_images(images["A"], 4, 4)  #test
    #image_ops.show_images(images["B"], 4, 4)  #test

    #image_ops.show_images(train["A"], 10, 10)  #test
    #image_ops.show_images(validate["A"], 10, 10)  #test

    #image_ops.show_images(train["B"], 4, 4)  #test
    #image_ops.show_images(validate["B"], 4, 4)  #test
    
    
    train_stack = train
    for i, key in enumerate(train):
        train_stack[key] = create_stack(train[key])
        image_ops.print_progress_bar(i, 25, prefix='Creating stack for letter {0}'.format(key),
                                     suffix='Complete', length=50)
    stack_means = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
                   "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
                   "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
                   "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
                   "Y": [], "Z": []}
    for i, val_key in enumerate(validate):
        for j, tr_key in enumerate(train_stack):
            stack_means[tr_key].append(validate_stack(train_stack[tr_key], validate[val_key]))
        image_ops.print_progress_bar(i, 25, prefix='Validating letters for {0}'.format(val_key),
                                     suffix='Complete', length=50)
    
    print(stack_means)
    '''
    
    keylen = images["A"].shape[0]
    
    binarized_images = {}
    binarized_images.fromkeys(images.keys(), [])
    train = {}
    train.fromkeys(images.keys(), [])
    validate = {}
    validate.fromkeys(images.keys(), [])
    
    for key in images:
        binarized_images[key] = images[key][:,:,:,0]
        for i in range(len(binarized_images[key])):
            binarized_images[key][i] = to_binary(binarized_images[key][i], 127)
        train[key] = binarized_images[key][0:int(keylen*0.8)]
        validate[key] = binarized_images[key][int(keylen*0.8):keylen]
    
    train_projection = []
    train_labels = []
    for key in train:
        for img in train[key]:
            train_projection.append(img)
            train_labels.append(key)
    
    val_projection = []
    val_labels = []
    for key in validate:
        for img in validate[key]:
            val_projection.append(img)
            val_labels.append(key)
    
    start = time.time()
    guessed_labels = projection(train_projection, val_projection, train_labels)
    end = time.time()
    
    print('Finished in ' + str(round(end - start)) + ' seconds')
    
    count = 0
    for i, guessed_label in enumerate(guessed_labels):
        image_ops.print_progress_bar(i, len(guessed_labels) - 1, 
                                     prefix='Checking predictions',
                                     suffix='Complete', length=50)
        if guessed_label == val_labels[i]: 
            count += 1
    
    print('')
    print('Results of projection: ')
    print(str(count) + '/' + str(len(val_labels)) + ' correct')
    print(str(count / len(guessed_labels) * 100) + '% accuracy')