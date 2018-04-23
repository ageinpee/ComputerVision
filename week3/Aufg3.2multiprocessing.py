# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:40:51 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from multiprocessing import Pool
import timeit


valImgs = np.array([])
valLabels = np.array([])

trImgs = np.array([])
trLabels = np.array([])

#Aufg. 2

d = np.load('./trainingsDatenFarbe2.npz')
trImgs = d['data']
trLabels = d['labels']
valD = np.load('./validierungsDatenFarbe2.npz')
valImgs = valD['data']
valLabels = valD['labels']

#Aufg. 1

def Aufg1():
    #1. Load Lenna.png
    lenna = imread("./Lenna.png")

    #2. Divide Lenna.png by colour-channels and put them together as a greyscale-image
    lennar = lenna[:,:,0]
    lennag = lenna[:,:,1]
    lennab = lenna[:,:,2]

    lenna_grey = lenna
    for pixel_x in range(lenna.shape[0]):
        for pixel_y in range(lenna.shape[1]):
            lenna_grey[pixel_x][pixel_y] = (int(lennar[pixel_x][pixel_y]) +
                                            int(lennag[pixel_x][pixel_y]) +
                                            int(lennab[pixel_x][pixel_y]))/3
    plt.imshow(lenna_grey, cmap='Greys_r')

    #5. Alle Werte werden invertiert
    lennainv = 255-lenna

    #3. heller -> farbe staerker vorhanden
    fig, ax = plt.subplots(2, 3)

    ax[0,0].imshow(lennar, cmap='Greys_r')
    ax[0,0].set_title('Rotkanal')

    ax[0,1].imshow(lennag, cmap='Greys_r')
    ax[0,1].set_title('Gruenkanal')

    ax[0,2].imshow(lennab, cmap='Greys_r')

    ax[0,2].set_title('Blaukanal')

    ax[1,0].imshow(lennainv)
    ax[1,0].set_title('RGB invertiert')

    ax[1,1].imshow(lenna)
    ax[1,1].set_title('Original')

    #4. Put the colour-channels together in a wrong order.
    """
    Wenn die Farbkanäle in falscher Reihenfolge zusammengesetzt werden, 
        also zum Beispiel der Rote und Grüne Farbkanal vertauscht, werden
        diese als die jeweiligen Farben interpretiert. Das beudeutet, die 
        Rot-Werte im neuen Bild entsprechen den Grün-Werten im alten Bild
        und die Grün-Werte im neuen Bild den Rot-Werten im alten Bild.
    """

    plt.show(block=True)  # This command shows the plt-windows. Not necessary in all IDEs
         # Opening dialog windows have to be closed before the program continues.


    #6. Calculate mean and standard deviation without dividing into colour-channels
    # --> Über die Methoden mean und std mit Angabe der Axen.
    rgbmean = np.mean(lenna, axis=(0,1))
    rgbstd = np.std(lenna, axis=(0,1))

def Aufg31Farbe():
    trMeans = []
    for count in range(trImgs.shape[0]):
        trMeans.append(np.mean(trImgs[count,:,:], axis = (0,1)))
    
    valMeans = []
    for count in range(valImgs.shape[0]):
        valMeans.append(np.mean(valImgs[count,:,:], axis = (0,1)))
        
    labels = []
    
    for count in range(len(valMeans)):
        dists = []    
        for i in range(len(trMeans)):
            dists.append(np.linalg.norm(trMeans[i]-valMeans[count]))
        labels.append(trLabels[dists.index(min(dists))])
    
    return labels


#50% Genauigkeit, also deutlich besseres Ergebnis

#Aufg. 3

def makeHist(img, nrBins):
    return np.histogram(img, bins = nrBins)[0]


def Aufg32Farbe(bins):
    global trImgs
    global trLabels
    global valImgs
    global valLabels

    trHists = []
    for count in range(trImgs.shape[0]):
        img = trImgs[count,:,:,:]
        rotKanal = img[:,:,0]
        gruenKanal = img[:,:,1]
        blauKanal = img [:,:,2]
        trHists.append((makeHist(rotKanal, bins), makeHist(blauKanal, bins), makeHist(gruenKanal, bins)))

    valHists = []
    for count in range(valImgs.shape[0]):
        img = valImgs[count,:,:,:]
        rotKanal = img[:,:,0]
        gruenKanal = img[:,:,1]
        blauKanal = img [:,:,2]
        valHists.append((makeHist(rotKanal, bins), makeHist(blauKanal, bins), makeHist(gruenKanal, bins)))

    labels = []
    for count in range(len(valHists)):
        dists = []
        for i in range(len(trHists)):
            dists.append(np.linalg.norm(np.asarray(trHists[i])-np.asarray(valHists[count])))
        labels.append(trLabels[dists.index(min(dists))])

    return labels
    #Optimales Ergebnis is 56.666...% mit 219 bins
    #Wenn ich mich richtig erinnere nicht viel besser als mit Graustufenhistogrammen


def print_misc():
    print 'Task 2: assign pictures to a group of pictures'
    print '----------------------------------------------------------------------------'
    print np.sum(valLabels==Aufg31Farbe()), 'Labels were chosen correct'
    print np.sum(valLabels==Aufg31Farbe()) / float(valImgs.shape[0]) * 100, '% were chosen correct'
    print '----------------------------------------------------------------------------'

    print 'Task 3: assign pictures to a group of pictures with the help of histograms'
    print '----------------------------------------------------------------------------'
    #print optimalBins()
    print np.sum(valLabels==Aufg32Farbe(219)), 'were chosen correct'
    print np.sum(valLabels==Aufg32Farbe(219)) / float(valImgs.shape[0]) * 100, '% were chosen correct'

if __name__ == '__main__':
    solutions = []
    Aufg1()
    print_misc()

    start_time = timeit.default_timer()

    p = Pool()
    solutions = p.map(Aufg32Farbe, range(1,256), chunksize=256/4)

    best_sol = solutions[0]
    mult_sols = []

    for sol in solutions:
        if np.sum(sol == valLabels) > np.sum(best_sol == valLabels):
            best_sol = sol
    for sol in solutions:
        if np.sum(sol == valLabels) == np.sum(best_sol == valLabels):
            mult_sols.append(sol)
            mult_sols.append(solutions.index(sol) + 1)

    print mult_sols
    print '--> optimal number of bins for the histograms'
    print 'Runtime of parallel code =', timeit.default_timer() - start_time


