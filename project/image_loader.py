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

import libarchive

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
    with libarchive.reader(file_path) as reader:
        for e in reader:
            # (The entry evaluates to a filename.)

            print("> %s" % (e))
    return files