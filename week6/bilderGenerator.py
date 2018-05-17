#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:32:28 2018

@author: wilms
"""

import numpy as np
np.random.seed(123)

def zieheBilder(anzBilder):
    """
    Returns
    -------
    x1 : Array der Shape (anzBilder) mit den Mittelwerten der generierten Bilder.
    
    x2 : Array der Shape (anzBilder) mit den Standardabweichungen der generierten Bilder.
    
    y : Array der Shape (anzBilder) mit Labels der generierten Bilder.
    
    Examples
    --------
    >>> mws, stds, labels = zieheBilder(20)
    >>> mws
    array([ 117.42570112,  160.10112923,  117.42317858,  163.11348509,
        114.00957359,  136.93796604,  139.27300204,  178.09387744,
        118.89808973,  166.23303558,  172.14561475,  119.28160907,
        100.32698356,  171.56721531,  172.01214607,  111.76648838,
        177.31218876,  102.00955277,  154.93706257,  147.3702592 ])
    >>> stds
    array([ 93.20951897,  49.24443909,  86.1456785 ,  22.67134231,
        87.81576576,  80.05527695,  76.55062801,  61.10255823,
        72.60596699,  80.25621081,  89.30357491,  87.59380439,
        67.18501442,  83.40392195,  49.70482912,  67.29974241,
        65.21536854,  87.60225817,  62.68756725,  56.00178831])
    >>> labels
    array([ 1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,
        1, -1, -1])
    """
    numSamples=int(anzBilder/2)
    mean1 = [120,80]
    cov1 = [[180,0],[0,60]]
    x1_1,x1_2 = np.random.multivariate_normal(mean1,cov1,numSamples).T
    y1 = [1]*numSamples
    mean2 = [160,60]
    cov2 = [[100,0],[0,160]]
    x2_1,x2_2 = np.random.multivariate_normal(mean2,cov2,numSamples).T
    y2 = [-1]*numSamples
    permutationIndices = np.random.permutation(range(numSamples*2))
    x1 = np.append(x1_1,x2_1)[permutationIndices]
    x2 = np.append(x1_2,x2_2)[permutationIndices]
    y = np.array(y1+y2)[permutationIndices]
    return x1,x2,y