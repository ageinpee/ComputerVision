# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:24:44 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879)
"""

import numpy as np
import matplotlib as plt
from skimage.io import imread, imshow
import glob
from skimage.filters import threshold_otsu
import itertools 
from skimage.measure import regionprops, label

val_imgs = glob.glob('./haribo1/hariboVal/*.png')
tr_imgs = glob.glob('./haribo1/hariboTrain/*.png')

val_labels = [x.split('/')[-1].split('.')[0].split('_')[0].split('\\')[1] for x in val_imgs]
tr_labels = [x.split('/')[-1].split('.')[0].split('_')[0].split('\\')[1] for x in tr_imgs]


# general helper functions
def load_imgs():
    for i,img in enumerate(val_imgs):
        val_imgs[i] = imread(img)
    for i,img in enumerate(tr_imgs):
        tr_imgs[i] = imread(img)


def create_seperator():
    sep = ''
    for i in range(15):
        sep += '---------'
    return sep


def compare_correct(labels):
    count = 0
    for i, pred_label in enumerate(labels):
        if val_labels[i] == pred_label:
            count += 1
    return count


def to_binary_hsv(img, lower, upper):
    return (lower < img) & (img < upper)


def to_binary_rgb(img, value):
    return img < value


def show_imgs_hsv(imgs, subplot_x, subplot_y):
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    for i in range(subplot_x):
        for j in range(subplot_y):
            if(imgs[i] is None):
                break
            else:
                ax[i, j].imshow(plt.colors.hsv_to_rgb(imgs[(i*subplot_y)+j]), 'Greys_r')
    plt.pyplot.show(block=True)


def show_imgs_rgb(imgs, subplot_x, subplot_y):
    fig, ax = plt.pyplot.subplots(subplot_x, subplot_y)

    for i in range(subplot_x):
        for j in range(subplot_y):
            if(imgs[i] is None):
                break
            else:
                ax[i, j].imshow(imgs[(i*subplot_y)+j], 'Greys_r')
    plt.pyplot.show(block=True)


def bbox(img):
    img = img.astype(np.int)
    props = regionprops(img)[0]
    return props.bbox #xMin, yMin, xMax, yMax -> index as (0,2),(1,3)

# task functions
def classify_means(val, tr):
    val_means = []
    tr_means = []
    
    for img in val:
        val_means.append(np.mean(img, axis = (0,1)))
        
    for img in tr:
        tr_means.append(np.mean(img, axis = (0,1)))
    
    return val_means, tr_means

def compare_means(val, tr):
    means = classify_means(val, tr)
    val_means = means[0]
    tr_means = means[1]
    labels = []
    
    for count in range(len(val_means)):
        dists = []
        for i in range(len(tr_means)):
            dists.append(np.linalg.norm(tr_means[i] - val_means[count]))
        labels.append(tr_labels[np.argmin(dists)])
    return labels
    
    
def classify_3dhists(val, tr):
    val_hists = []
    tr_hists = []
    for img in val:
        img = img.reshape((img.shape[0]*img.shape[1],3))
        val_hists.append(np.histogramdd(img, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0])
    for img in tr:
        img = img.reshape((img.shape[0]*img.shape[1],3))
        tr_hists.append(np.histogramdd(img, bins = [8,8,8], range=((0,256),(0,256),(0,256)))[0])
    return val_hists, tr_hists


def compare_3dhists(val, tr):
    hists = classify_3dhists(val, tr)
    val_hists = hists[0]
    tr_hists = hists[1]
    labels = []
    
    for count in range(len(val_hists)):
        dists = []
        for i in range(len(tr_hists)):
            dists.append(np.linalg.norm(tr_hists[i] - val_hists[count]))
        labels.append(tr_labels[np.argmin(dists)])

    return labels


def convert_to_hsv():
    val_imgs_hsv = map(plt.colors.rgb_to_hsv, val_imgs)
    tr_imgs_hsv = map(plt.colors.rgb_to_hsv, tr_imgs)
    return val_imgs_hsv, tr_imgs_hsv


def binarize():    
    val_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in val_imgs]))
    tr_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in tr_imgs]))

    val_binary = list(itertools.starmap(to_binary_hsv, [(img, 30, 110) for img in val_imgs_greys]))
    tr_binary = list(itertools.starmap(to_binary_hsv, [(img, 30, 110) for img in tr_imgs_greys]))
    val_binary = [255*img for img in val_binary]
    tr_binary = [255*img for img in tr_binary]

    return val_binary, tr_binary


def binarize_otsu():
    val_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in val_imgs]))
    tr_imgs_greys = list(itertools.starmap(np.dot, [(img, [0.299, 0.587, 0.114]) for img in tr_imgs]))
    val_otsu = map(threshold_otsu, val_imgs_greys)
    tr_otsu = map(threshold_otsu, tr_imgs_greys)

    val_binary_otsu = list(itertools.starmap(to_binary_rgb,
                                             [(img, val_otsu[i]) for i,img in enumerate(val_imgs_greys)]))
    tr_binary_otsu = list(itertools.starmap(to_binary_rgb,
                                            [(img, tr_otsu[i]) for i,img in enumerate(tr_imgs_greys)]))
    val_binary_otsu = [255*img for img in val_binary_otsu]
    tr_binary_otsu = [255*img for img in tr_binary_otsu]

    return val_binary_otsu, tr_binary_otsu


def create_bounding_boxes(method = 'fixed'):
    if method == 'fixed':
        binarys = binarize()
    else: 
        binarys = binarize_otsu()
        
    val_binarys = binarys[0]
    tr_binarys = binarys[1]

    tr_boxes = []
    for img in tr_binarys:
        tr_boxes.append(bbox(img))
    
    val_boxes = []
    for img in val_binarys:
        val_boxes.append(bbox(img))

    tr_imgs_boxed =[]
    for i,img in enumerate(tr_imgs):
        tr_imgs_boxed.append(img[tr_boxes[i][0]:tr_boxes[i][2], tr_boxes[i][1]:tr_boxes[i][3], :])
        
    val_imgs_boxed = []
    for i, img in enumerate(val_imgs):
        val_imgs_boxed.append(img[val_boxes[i][0]:val_boxes[i][2], val_boxes[i][1]:val_boxes[i][3], :])

    return val_imgs_boxed, tr_imgs_boxed

#Sollte funktionieren, aber bei hamburg2.png dauert es bei mir so lange, dass ich kein
#Ergebnis bekomme (bei paar minuten warten)
def regiongrowing():
    img = imread('./hamburg2.png')
    img = img[:,:,0]*0.33+img[:,:,1]*0.33+img[:,:,2]*0.33
    queue = []
    value = img[0,0]
    print value
    binarized = np.ones(img.shape)
    binarized = binarized*255
    binarized[0,0] = 0
    queue.append((0,0))
    finished = []
    while len(queue) > 0:
        px = queue[0]
        binarized[px[0], px[1]] = 0
        for pos in neighbors(px[0], px[1], img.shape):
            if img[pos[0], pos[1]] == value:
                binarized[pos[0], pos[1]] = 0
                if not pos in finished:
                    queue.append(pos)
                finished.append(pos)
        queue.pop(0)
        #imshow('progress', binarized)
        #print binarized
        
        #raw_input('press enter')
    return binarized
                
#geklaute Funktion!
def neighbors(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out    


if __name__ == '__main__':
    #load_imgs()
    
    #print 'Classification with entire image:\n'
    #print val_labels
    #print compare_means(val_imgs, tr_imgs)
    #print compare_correct(compare_means(val_imgs, tr_imgs)), 'Labels were chosen correctly (Descriptor: mean)'
    #print compare_3dhists(val_imgs, tr_imgs)
    #print compare_correct(compare_3dhists(val_imgs, tr_imgs)), 'Labels were chosen correctly (Descriptor: 3DHist)'
    #print create_seperator()
    
    #print 'Classification with bounding boxes:\n'
    #print val_labels
    #imgs_boxed = create_bounding_boxes()
    #val_imgs_boxed = imgs_boxed[0]
    #tr_imgs_boxed = imgs_boxed[1]
    #print compare_means(val_imgs_boxed, tr_imgs_boxed)
    #print compare_correct(compare_means(val_imgs_boxed, tr_imgs_boxed)), 'Labels were chosen correctly (Descriptor: mean)'
    #print compare_3dhists(val_imgs_boxed, tr_imgs_boxed)
    #print compare_correct(compare_3dhists(val_imgs_boxed, tr_imgs_boxed)), 'Labels were chosen correctly (Descriptor: 3DHist)'
    #print create_seperator()
    
    #Die Ergebnisse beim Klassifizieren mit den Bildausschnitten sind erheblich besser:
    #schon der Mittelwert zeigt eine Verbesserung von ~17% auf ~67% Genauigkeit,
    #was erheblich besser als eine Zufallsauswahl ist. Diese Verbesserung ist auf
    #die Elimination des Hintergrundes zurueckzufuehren, der den Mittelwert natuerlich
    #deutlich beeinflusst weil er die groesste Flaeche darstellt.
    
    #Bei der Klassifizierung mit 3DHistogrammen ist die Verbesserung noch deutlicher:
    #von 25% (etwas under dem Zufallswert) auf ~92%! Auch hier werden durch Entfernen des
    #Hintergrunds die Unterschiede der Bilder mehr hervorgehoben. 
    #Interessanterweise gab es zwischen dem Otsu-Verfahren und dem manuellen Schwellwert
    #keinen Unterschied bei der Genauigkeit, obwohl das Otsu-Verfahren bei manchen
    #Bilder deutlich mehr Hintergrund mit einbezieht.
    
    
    #show_imgs_rgb(create_bounding_boxes('otsu')[1], 3, 13)
    imshow(regiongrowing())