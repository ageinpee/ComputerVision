# -*- coding: utf-8 -*-
import numpy as np
from scipy import io
import scipy.ndimage.interpolation
import image_ops

np.random.seed(4505918)

emnist = io.loadmat(input("Enter filepath for EMNIST data: "))

#Trainingsdaten werden aus dem dictionary geladen und von shape 784 auf 28x28 umgeformt
imgs = emnist['dataset'][0][0][0][0][0][0]
imgs = imgs.reshape(imgs.shape[0], 1, 28, 28, order="A")

#Labels werden geladen und die relevanted Klassen "A" - "Z" herausgefiltert
labels = emnist['dataset'][0][0][0][0][0][1]
ind = np.where((labels > 9) & (labels < 36))[0]

trlabels = []
trimgs = []

#Die Daten werden in geordneten Listen gespeichert
for index in ind:
    trlabels.append(labels[index][0])
    trimgs.append(imgs[index])

#Dasselbe passiert nun für die Testdaten
imgs = emnist['dataset'][0][0][1][0][0][0]
imgs = imgs.reshape(imgs.shape[0], 1, 28, 28, order="A")

labels = emnist['dataset'][0][0][1][0][0][1]
ind = np.where((labels > 9) & (labels < 36))[0]

testlabels = []
testimgs = []

for index in ind:
    testlabels.append(labels[index][0])
    testimgs.append(imgs[index])

#Gibt jedes einzigartige label einmal aus
unique, testcounts = np.unique(testlabels, return_counts=True)

letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}

#Daten werden in Dictionary als 28x28 RGBA Bilder gespeichtert, keys sind die Klassen
for char in unique:
    ind = np.where(trlabels == char)[0]
    for index in ind:
        a = trimgs[index][0]
        b = np.full(shape=(28,28,1), fill_value=255, dtype=np.uint8)
        c = np.dstack((a,a,a,b))
        letters[chr(char + 55)].append(c)

#Es werden 2000 Daten pro Klasse zufällig entnommen
for key in letters:
    letters[key] = np.random.permutation(letters[key])[:2000]

#Die Daten werden als npz gespeichert
for key in letters:
        image_ops.save_images_npz('./EMNIST_2K_TRAINING/Data_' + key, letters[key])

#Dieselben Operationen für die Testdaten
letters = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [],
               "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
               "M": [], "N": [], "O": [], "P": [], "Q": [], "R": [],
               "S": [], "T": [], "U": [], "V": [], "W": [], "X": [],
               "Y": [], "Z": []}

for char in unique:
    ind = np.where(testlabels == char)[0]
    for index in ind:
        a = testimgs[index][0]
        b = np.full(shape=(28,28,1), fill_value=255, dtype=np.uint8)
        c = np.dstack((a,a,a,b))
        letters[chr(char + 55)].append(c)

#"K wird dabei von 384 auf 400 augmentiert durch Rotation
for i in range(18):
    letters["K"].append(scipy.ndimage.interpolation.rotate(letters["K"][i], 
           float(np.random.choice([-10, -5, 5, 10])), 
           reshape=False, mode='nearest'))

#Es werden 400 Daten pro Klasse zufällig entnommen
for key in letters:
    letters[key] = np.random.permutation(letters[key])[:400]

#Die Daten werden als npz gespeichert
for key in letters:
        image_ops.save_images_npz('./EMNIST_400_TEST/Data_' + key, letters[key])