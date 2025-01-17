# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi(6989075)

This file is supposed to handle basic image i/o operations for the project.
    This includes loading and showing images.
"""

"""
Imports:
------------------------------------------------------------------------------------------------------------------------
"""
import os
import random
import itertools

import numpy as np
import skimage.io
import matplotlib as plt
import scipy.ndimage.interpolation


"""
Helping functions:
------------------------------------------------------------------------------------------------------------------------
"""

# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int((length * iteration) / total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print ('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=" ")
    # '\r{0} |{1}| {2} {3}'.format(prefix, bar, percent, suffix),
    # Print New Line on Complete
    if iteration == total:
        print()


def remove_array(L, arr):
    if isinstance(L, (np.ndarray, np.generic)):
        np.setdiff1d(L, arr)
    else:
        ind = 0
        size = len(L)
        while ind != size and not np.array_equal(L[ind], arr):
            ind += 1
        if ind != size:
            L.pop(ind)
        else:
            raise ValueError('array not found in list.')


"""
Main functions:
------------------------------------------------------------------------------------------------------------------------
"""


def augment_images(images, wanted_length, label='unknown'):
    """
    Augmentiert einen Datensatz von Bildern.
    @params
        images --> Datensatz der Bilder.
        wanted_length --> die gewünschte Anzahl der Bilder pro Klasse
        label --> das Label der Klasse. Default='unknown'
    """
    out_images = []
    if len(images) >= wanted_length:
        print_progress_bar(0, wanted_length, prefix='augmenting images for label {0}:'.format(label), suffix='Complete', length=50)
        if len(images) == wanted_length:
            print_progress_bar(wanted_length, wanted_length, prefix='augmenting images for label {0}:'.format(label), suffix='Complete', length=50)
            print('')
            return images

        for i in range(wanted_length):
            choice = random.choice(images)
            remove_array(images, choice)
            out_images.append(choice)
            print_progress_bar(i, wanted_length, prefix='augmenting images for label {0}:'.format(label), suffix='Complete', length=50)
    else:
        if isinstance(images, (np.ndarray, np.generic)):
            for i in range(images.shape[0]):
                out_images.append(images[i])
        else:
            out_images = images
        print_progress_bar(0, wanted_length-len(images), prefix='augmenting images for label {0}:'.format(label), suffix='Complete', length=50)
        for i in range(wanted_length-len(images)):
            choice = random.choice(images)
            choice = scipy.ndimage.interpolation.rotate(choice, float(random.choice([-10, -8.5, -7, -6, -5, -2,
                                                                                     2, 5, 6, 7, 8.5, 10])), reshape=False, mode='constant', cval=255)
            out_images.append(choice)
            #print_progress_bar(i, wanted_length-len(images), prefix='augmenting images for label {0}:'.format(label), suffix='Complete', length=50)
    print('')
    return out_images



def show_images(imgs, subplot_x, subplot_y):
    """
    Hilfmethode zur Ausgabe von mehreren Bildern auf einmal.
    @params:
        imgs --> Liste oder ndarray mit Bildern.
        subplot_x --> Anzahl der max. Bilderanzahl auf der x-Achse
        subplot_y --> Anzahl der max. Bilderanzahl auf der y-Achse
    """
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    if len(imgs) > subplot_x * subplot_y:
        print("WARNING: there are more images than there is space in the plot to show them. "
              "Therefor only the first", subplot_x*subplot_y, "images are shown.")
    index_count = 0
    for i in range(subplot_x):
        for j in range(subplot_y):
            if index_count >= len(imgs):
                ax[i,j].set_axis_off()
            else:
                ax[i, j].set_axis_off()
                ax[i, j].imshow(imgs[(i*subplot_y)+j], 'gray_r')
            index_count += 1
    plt.pyplot.show(block=True)


"""
loads .png images from a directory and returns them as a dict. Each key holds a list of ndarrays
"""
def load_images(file_path):
    letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}
    #tempcount = 0   # <<<<<<<<<<<<<<<<<<<<<<<<<
    length = len(os.listdir(file_path))
    print_progress_bar(0, length, prefix='loading images:', suffix='Complete', length=50)
    for i, filename in enumerate(os.listdir(file_path)):
        if filename.endswith(".png"):
            first, second, third = filename.split("_")
            if second == "65":
                letters["A"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "66":
                letters["B"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "67":
                letters["C"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "68":
                letters["D"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "69":
                letters["E"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "70":
                letters["F"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "71":
                letters["G"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "72":
                letters["H"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "73":
                letters["I"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "74":
                letters["J"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "75":
                letters["K"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "76":
                letters["L"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "77":
                letters["M"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "78":
                letters["N"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "79":
                letters["O"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "80":
                letters["P"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "81":
                letters["Q"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "82":
                letters["R"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "83":
                letters["S"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "84":
                letters["T"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "85":
                letters["U"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "86":
                letters["V"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "87":
                letters["W"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "88":
                letters["X"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "89":
                letters["Y"].append(skimage.io.imread(file_path + "/" + filename))
            elif second == "90":
                letters["Z"].append(skimage.io.imread(file_path + "/" + filename))

        print_progress_bar(i+1, length, prefix='Progress:', suffix='Complete', length=50)
        #tempcount += 1  # <<<<<<<<<<<<<<<<<<<<<<<<<
        #if tempcount == 20:     # <<<<<<<<<<<<<<<<<<<<<<<<<
         #   break   # <<<<<<<<<<<<<<<<<<<<<<<<<
    return letters


"""
loads .npz image-arrays from a directory and returns them as a dict. Each key holds a ndarray of ndarrays.
"""
def load_images_npz(file_path):
    letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}
    for i, filename in enumerate(os.listdir(file_path)):
        if filename.endswith(".npz"):
            label = filename.split("_")[1].split(".")[0]
            data = np.load("{0}/{1}".format(file_path, filename))
            letters[label] = data.f.arr_0
    return letters


"""
saves a list of ndarrays as a .npz compressed-numpy-file
"""
def save_images_npz(file_path, images):
    np.savez_compressed(file_path, images)


def test():
    images = load_images(input("Enter a file path for the images: "))
    aug_imgs = {}
    aug_imgs.fromkeys(images.keys(), [])
    for key in images:
        aug_imgs[key] = augment_images(images[key], 2000, label=key)
    show_images(aug_imgs["F"][1000:2000], 30, 30)
    for key in aug_imgs:
        save_images_npz('C:/Users/Moritz Lahann/Desktop/STUDIUM/PRAKTIKUM COMPUTERVISION/DATA/TEST/' + key, aug_imgs[key])
