# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi

This file is supposed to handle basic image i/o operations for the project.
    This includes loading and showing images.
"""

"""
Imports:
--------------------------------------------------------------------------
"""
import os

import skimage.io
import matplotlib as plt


"""
Helping functions:
--------------------------------------------------------------------------
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
    bar = fill * filled_length + '--' * (length - filled_length)
    print ('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=" ")
    # '\r{0} |{1}| {2} {3}'.format(prefix, bar, percent, suffix),
    # Print New Line on Complete
    if iteration == total:
        print()


"""
Main functions:
--------------------------------------------------------------------------
"""


def show_images(imgs, subplot_x, subplot_y):
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    if len(imgs) > subplot_x * subplot_y:
        print("WARNING: there are more images than there is space in the plot to show them. "
              "Therefor only the first", subplot_x*subplot_y, "images are shown.")
    for i in range(subplot_x):
        for j in range(subplot_y):
            if i * j == len(imgs):
                break
            else:
                ax[i, j].set_axis_off()
                ax[i, j].imshow(imgs[(i*subplot_y)+j], 'Greys_r')
    plt.pyplot.show(block=True)


def get_images(file_path):
    letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}
    #tempcount = 0   # <<<<<<<<<<<<<<<<<<<<<<<<<
    length = len(os.listdir(file_path))
    print_progress_bar(0, length, prefix='Progress:', suffix='Complete', length=50)
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
