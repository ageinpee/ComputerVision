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
import random

import numpy as np
import matplotlib.pyplot as plt

import image_ops


"""
Helping functions:
------------------------------------------------------------------------------------------------------------------------
"""


def to_binary(img, value):
    return img > value


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
    stack = np.zeros(imgs[0].shape[0], imgs[0].shape[1])
    for i, img in enumerate(imgs):
        stack = (stack + img)
    stack = stack / len(imgs)
    return stack


def validate_stack(stack, imgs):
    diff_list = []
    for i, img in enumerate(imgs):
        diff_list.append(np.mean(stack - img))
    return diff_list


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
    train = images
    validate = images
    for key in images:
        print(key, len(images[key]))
        images[key] = to_binary(images[key], 127)*255   # binarization of all images.
        train[key], validate[key] = create_tr_val_data(images[key], 180, 20)   # no parameters = standard of 800/200 tr/val
                                                                      # all tr/val lists are still ordered by label

    for key in train:
        print("train", key, len(train[key]))
        print("validate", key, len(validate[key]))
    image_ops.show_images(images["A"], 4, 4)  #test
    image_ops.show_images(images["B"], 4, 4)  #test

    image_ops.show_images(train["A"], 10, 10)  #test
    image_ops.show_images(validate["A"], 10, 10)  #test

    image_ops.show_images(train["B"], 4, 4)  #test
    image_ops.show_images(validate["B"], 4, 4)  #test
