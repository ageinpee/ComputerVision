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
import imageio
import os

"""
Helping functions:
--------------------------------------------------------------------------
"""


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

"""
Main functions:
--------------------------------------------------------------------------
"""


def get_images(file_path):
    files = []
    letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}
    #tempcount = 0   # <<<<<<<<<<<<<<<<<<<<<<<<<
    length = len([os.listdir(file_path)])
    printProgressBar(0, length, prefix='Progress:', suffix='Complete', length=50)
    for i, filename in enumerate(os.listdir(file_path)):
        if filename.endswith(".png"):
            first, second, third = filename.split("_")
            if second == "65":
                letters["A"].append(imageio.imread(file_path + "/" + filename))
            elif second == "66":
                letters["B"].append(imageio.imread(file_path + "/" + filename))
            elif second == "67":
                letters["C"].append(imageio.imread(file_path + "/" + filename))
            elif second == "68":
                letters["D"].append(imageio.imread(file_path + "/" + filename))
            elif second == "69":
                letters["E"].append(imageio.imread(file_path + "/" + filename))
            elif second == "70":
                letters["F"].append(imageio.imread(file_path + "/" + filename))
            elif second == "71":
                letters["G"].append(imageio.imread(file_path + "/" + filename))
            elif second == "72":
                letters["H"].append(imageio.imread(file_path + "/" + filename))
            elif second == "73":
                letters["I"].append(imageio.imread(file_path + "/" + filename))
            elif second == "74":
                letters["J"].append(imageio.imread(file_path + "/" + filename))
            elif second == "75":
                letters["K"].append(imageio.imread(file_path + "/" + filename))
            elif second == "76":
                letters["L"].append(imageio.imread(file_path + "/" + filename))
            elif second == "77":
                letters["M"].append(imageio.imread(file_path + "/" + filename))
            elif second == "78":
                letters["N"].append(imageio.imread(file_path + "/" + filename))
            elif second == "79":
                letters["O"].append(imageio.imread(file_path + "/" + filename))
            elif second == "80":
                letters["P"].append(imageio.imread(file_path + "/" + filename))
            elif second == "81":
                letters["Q"].append(imageio.imread(file_path + "/" + filename))
            elif second == "82":
                letters["R"].append(imageio.imread(file_path + "/" + filename))
            elif second == "83":
                letters["S"].append(imageio.imread(file_path + "/" + filename))
            elif second == "84":
                letters["T"].append(imageio.imread(file_path + "/" + filename))
            elif second == "85":
                letters["U"].append(imageio.imread(file_path + "/" + filename))
            elif second == "86":
                letters["V"].append(imageio.imread(file_path + "/" + filename))
            elif second == "87":
                letters["W"].append(imageio.imread(file_path + "/" + filename))
            elif second == "88":
                letters["X"].append(imageio.imread(file_path + "/" + filename))
            elif second == "89":
                letters["Y"].append(imageio.imread(file_path + "/" + filename))
            elif second == "90":
                letters["Z"].append(imageio.imread(file_path + "/" + filename))

        if i % 100 == 0:
            printProgressBar(i+1, length, prefix='Progress:', suffix='Complete', length=50)
        #tempcount += 1  # <<<<<<<<<<<<<<<<<<<<<<<<<
        #if tempcount == 20:     # <<<<<<<<<<<<<<<<<<<<<<<<<
         #   break   # <<<<<<<<<<<<<<<<<<<<<<<<<
    return letters
