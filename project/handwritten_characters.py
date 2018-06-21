# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi(6989075)

This file is supposed to host the script for a classification problem approach.
    The problem is to classify handwritten letters. For solving the problem different
    image augumentations will be used.
"""

"""
Imports:
--------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

import image_ops


"""
Helping functions:
--------------------------------------------------------------------------
"""


def to_binary(img, value):
    return img < value


"""
Main functions:
--------------------------------------------------------------------------
"""


"""
Main execution
--------------------------------------------------------------------------
"""

if __name__ == '__main__':
    images = image_ops.load_images_npz(input("Enter a file path for the data: "))

    for key in images:
        print(key, len(images[key]))

    for key in images:
        images[key] = image_ops.augment_images(images[key], 1000, key)
        image_ops.save_images_npz("data/Data_Augmented_cShape/Data_" + key, images[key])

    image_ops.show_images(images["A"], 20, 20)  #test
    image_ops.show_images(images["B"], 20, 20)  #test