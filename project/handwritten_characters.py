# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)

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

import image_loader


"""
Helping functions:
--------------------------------------------------------------------------
"""


"""
Main functions:
--------------------------------------------------------------------------
"""


"""
Main execution
--------------------------------------------------------------------------
"""

if __name__ == '__main__':
    images = image_loader.get_images("data/Data_Processed")
    plt.imshow(images["A"][0])  # test
    plt.show()
