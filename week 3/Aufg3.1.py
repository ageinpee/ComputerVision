# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:54:10 2018

@author: 6peters
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

d = np.load('./trainingsDaten2.npz')
trImgs = d['data']
trLabels = d['labels']

valD = np.load('./validierungsDaten2.npz')
valImgs = valD['data']
valLabels = valD['labels']    



def Aufg1():
    trMeans = []
    for count in range(trImgs.shape[0]):
        trMeans.append(np.mean(trImgs[count,:,:]))
        
    valMeans = []
    for count in range(valImgs.shape[0]):
        valMeans.append(np.mean(valImgs[count,:,:]))
    
    labels = []
    
    for count in range(len(valMeans)):
        dists = []    
        for i in range(len(trMeans)):
            dists.append(np.linalg.norm(trMeans[i]-valMeans[count]))
        labels.append(trLabels[dists.index(min(dists))])
    
    return labels   

print Aufg1()
print np.sum(valLabels==Aufg1())
print np.sum(valLabels==Aufg1()) / float(valImgs.shape[0]) * 100


# Ziel: Erstmal besser als zufällige Zuweisung

#Aufg2

trHists = []
valHists = []

def hist_L2(hist1, hist2):
    return np.linalg.norm(hist1-hist2)  
    
for count in range(trImgs.shape[0]):
    trHists.append(np.histogram(trImgs[count,:,:], bins = 'sqrt')[0])

for count in range(valImgs.shape[0]):
    valHists.append(np.histogram(valImgs[count,:,:], bins = 'sqrt')[0])

def Aufg2():
        
    labels = []
    
    for count in range(len(valHists)):
        dists = []    
        for i in range(len(trHists)):
            dists.append(hist_L2(valHists[count], trHists[i]))
        labels.append(trLabels[dists.index(min(dists))])
    
    return labels
    
    
print np.sum(valLabels==Aufg2())
print np.sum(valLabels==Aufg2()) / float(valImgs.shape[0]) * 100

        
#Aufg3

def getWerte(i, Aufg):
    if Aufg == 1: labels = Aufg1()
    if Aufg == 2: labels = Aufg2()
    auto = 0
    hirsch = 0
    schiff = 0
    for count in range(len(valLabels)):
        if valLabels[count] == i:
            if labels[count] == 1:
                auto+=1
            elif labels[count] == 4:
                hirsch+=1
            else:
                schiff+=1
    return [auto, hirsch, schiff]

print "Vorhersagen: Auto, Hirsch, Schiff"
print "Eigentlich Auto:  ", getWerte(1,1)
print "Eigentlich Hirsch:", getWerte(4,1)
print "Eigentlich Schiff:", getWerte(8,1)        

print "Vorhersagen: Auto, Hirsch, Schiff"
print "Eigentlich Auto:  ", getWerte(1,2)
print "Eigentlich Hirsch:", getWerte(4,2)
print "Eigentlich Schiff:", getWerte(8,2)        
