# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)

This file is supposed to load the training and evaluation images from the .7z file
"""

"""
Imports:
--------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio

import os
import zipfile

"""
Helping functions:
--------------------------------------------------------------------------
"""



"""
Main functions:
--------------------------------------------------------------------------
"""


def get_images(file_path):
    files = []
    tempcount = 0 #<<<<<
    """
    with zipfile.ZipFile(file_path, "r") as zf:
        for filename in zf.namelist():
            basename,extension = splitext(file)
            if extension == '.png':
                print("reading file > " + filename)
                files.append(imageio.imread("data/Data_Processed.zip/" + filename))
            tempcount += 1 #<<<<<
            if tempcount == 20: #<<<<
                break #<<<<<
                """
    print(file_path)
    zf = zipfile.ZipFile(file_path, 'r')
    for filename in zf.namelist():
        basename, extension = os.path.splitext(filename)
        if extension == '.png':
            print("reading file > " + filename)
            files.append(imageio.imread(file_path + "/" + filename))
    return files #return imageio.mimread(file_path + "/Data_Processed")