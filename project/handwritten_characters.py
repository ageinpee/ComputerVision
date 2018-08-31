# -*- coding: utf-8 -*-
"""
Created on Mon June  11 12:48:00 2018

@author: Moritz Lahann(6948050), Henrik Peters(6945965), Michael Huang(6947879), Iman Maiwandi(6989075)

This file is supposed to host the script for a classification problem approach.
    The problem is to classify handwritten letters. For solving the problem different
    image augmentations will be used.
"""

"""
Imports:
------------------------------------------------------------------------------------------------------------------------
"""
#System:
import random

#Scientific
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import itertools
import keras
from sklearn.neighbors import KNeighborsClassifier



#Local
import image_ops

#Miscellaneous
import time


"""
Helping functions:
------------------------------------------------------------------------------------------------------------------------
"""

def to_binary(img, value):
    return img < value

def crop_image(img,tol=255):
    mask = img < tol
    coords = np.argwhere(mask)

    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    cropped = img[x0:x1, y0:y1]

    return cropped

def histogram_x(img):
    x_hist = []
    for i in img:
        x_hist.append(sum(i))
    return x_hist

def histogram_y(img):
    y_hist = []
    for i in np.transpose(img):
        y_hist.append(sum(i))
    return y_hist

"""
IMPORTANT NOTE: Input is either a list or ndarray. Output is a tuple of 2 lists of images
"""
def create_tr_val_data(imgs, num_tr=800, num_val=200):
    tr = []
    val = []
    for i in range(0, num_tr):
        choice = random.choice(range(len(imgs)))
        tr.append(imgs[choice])
        image_ops.remove_array(imgs, imgs[choice])
    for i in range(0, num_val):
        choice = random.choice(range(len(imgs)))
        val.append(imgs[choice])
        image_ops.remove_array(imgs, imgs[choice])
    return tr, val


"""
Main functions:
------------------------------------------------------------------------------------------------------------------------
"""


"""
takes a list of images as input and creates a image stack as described in ideas.txt
"""

def projection_preprocessing(images):
    binarized_images = {}
    binarized_images.fromkeys(images.keys(), [])
    
    for key in images:
        binarized_images[key] = images[key][:,:,:,0]
        for i in range(len(binarized_images[key])):
            binarized_images[key][i] = to_binary(binarized_images[key][i], 127)
    
    return binarized_images
    

def create_stack(imgs):
    cropped = []
    for img in imgs:
        #plt.imshow(img, 'gray_r')
        #plt.show(block=True)
        crop = crop_image(img)
        crop = skt.resize(crop, (28, 28, 4), anti_aliasing=True)
        #plt.imshow(crop, 'gray_r')
        #plt.show(block=True)
        cropped.append(crop)

    stack = sum(cropped)
    stack = stack / len(imgs)
    return stack


def validate_image(stack, img):
    img = np.asarray(img)
    crop = crop_image(img)
    crop = skt.resize(crop, (28, 28, 4), anti_aliasing=True)

    return abs(np.mean(stack - crop))


def image_stack(tr, val, labels):
    # val = list of arrays/lists. 26 elems. each list/array = validation images
    # labels = list of labels. 10400 elems (2k data). ordered corresponding to val.

    stacks = {}     # dict of all stacks. key defines which stack
    stacks.fromkeys(tr.keys(), [])
    for i, k in enumerate(tr):
        stacks[k] = create_stack(tr[k])
        image_ops.print_progress_bar(i, 25, prefix='Creating stack for letter {0}'.format(k),
                                     suffix='Complete', length=50)

    #stack_list = []                                              # nur für
    #for stack in stacks:                                         # das anzeigen
    #    stack_list.append((stacks[stack]*255).astype(np.uint8))  # der imagestacks
    #image_ops.show_images(stack_list, 5, 6)                      # zuständig

    means = []  # list of lists. 10400 elem (2k data). each list contains the means values from the
                # validation. So the shape is (10400, 26)
    subcount = 0
    for letters in val:
        for img in letters:
            mean_list = []
            for stack in stacks:
                mean_list.append(validate_image(stacks[stack], img))
            means.append(mean_list)
        image_ops.print_progress_bar(subcount, 26,
                                     prefix='Validating {0} images'.format(len(val)*len(letters)),
                                     suffix='Complete', length=50)
        subcount += 1


    computed_labels = []
    for ix, img_means in enumerate(means):
        comp = min(img_means)
        index = img_means.index(comp)
        if index == 0: computed_labels.append('A')
        elif index == 1: computed_labels.append('B')
        elif index == 2: computed_labels.append('C')
        elif index == 3: computed_labels.append('D')
        elif index == 4: computed_labels.append('E')
        elif index == 5: computed_labels.append('F')
        elif index == 6: computed_labels.append('G')
        elif index == 7: computed_labels.append('H')
        elif index == 8: computed_labels.append('I')
        elif index == 9: computed_labels.append('J')
        elif index == 10: computed_labels.append('K')
        elif index == 11: computed_labels.append('L')
        elif index == 12: computed_labels.append('M')
        elif index == 13: computed_labels.append('N')
        elif index == 14: computed_labels.append('O')
        elif index == 15: computed_labels.append('P')
        elif index == 16: computed_labels.append('Q')
        elif index == 17: computed_labels.append('R')
        elif index == 18: computed_labels.append('S')
        elif index == 19: computed_labels.append('T')
        elif index == 20: computed_labels.append('U')
        elif index == 21: computed_labels.append('V')
        elif index == 22: computed_labels.append('W')
        elif index == 23: computed_labels.append('X')
        elif index == 24: computed_labels.append('Y')
        elif index == 25: computed_labels.append('Z')
        image_ops.print_progress_bar(ix, len(val), prefix='Computing performance of image_stack',
                                     suffix='Complete', length=50)

    percent = 0
    for j in range(len(computed_labels)):
        if computed_labels[j] == labels[j]:
            percent += 1
    
    return 'Correct: ', percent, 'Total: ', len(computed_labels), 'Percent: ', percent/len(computed_labels), list(zip(computed_labels, labels))


def projection(tr_imgs, val_imgs, tr_labels):
    tr_x_hists = []
    tr_y_hists = []
    j = 0
    for img in tr_imgs:
        image_ops.print_progress_bar(j, len(tr_imgs) - 1, 
                                     prefix='Processing training images',
                                     suffix='Complete', length=50)
        tr_x_hists.append(histogram_x(img))
        tr_y_hists.append(histogram_y(img))
        j += 1
    
    val_x_hists = []
    val_y_hists = []
    j = 0
    for img in val_imgs:
        image_ops.print_progress_bar(j, len(val_imgs) - 1, 
                                     prefix='Processing validation images',
                                     suffix='Complete', length=50)
        val_x_hists.append(histogram_x(img))
        val_y_hists.append(histogram_y(img))
        j += 1
    
    return validate_projection(np.hstack((tr_x_hists, tr_y_hists)), np.hstack((val_x_hists, val_y_hists)), tr_labels)
    

def validate_projection(tr, val, tr_xy_labels):
    labels = []
    j = 0
    for count in range(len(val)):
        start = time.perf_counter()

        dists = []
        for i in range(len(tr)):
            dists.append(np.linalg.norm(tr[i] - val[count]))
        labels.append(tr_xy_labels[np.argmin(dists)])
        
        end = time.perf_counter()
        #note this is pretty inaccurate, maybe do for every X iterations
        remaining = (end - start) * (len(val) - j)
        image_ops.print_progress_bar(j, len(val) - 1, 
                                     prefix='Assigning labels',
                                     suffix='{0}s remaining'.format(round(remaining)), length=50)
        j += 1
    
    return labels

def sklearn_knn(tr_imgs, val_imgs, tr_labels, val_labels, nneighbors):  
    tr_x_hists = []
    tr_y_hists = []
    j = 0
    for img in tr_imgs:
        image_ops.print_progress_bar(j, len(tr_imgs) - 1, 
                                     prefix='Processing training images',
                                     suffix='Complete', length=50)
        tr_x_hists.append(histogram_x(img))
        tr_y_hists.append(histogram_y(img))
        j += 1
    
    val_x_hists = []
    val_y_hists = []
    j = 0
    for img in val_imgs:
        image_ops.print_progress_bar(j, len(val_imgs) - 1, 
                                     prefix='Processing validation images',
                                     suffix='Complete', length=50)
        val_x_hists.append(histogram_x(img))
        val_y_hists.append(histogram_y(img))
        j += 1
    
    for i in range(len(tr_labels)):
            tr_labels[i] = ord(tr_labels[i])-65
            
    for i in range(len(val_labels)):
            val_labels[i] = ord(val_labels[i])-65
    
    Y_train = keras.utils.to_categorical(tr_labels, 26)
    Y_val = keras.utils.to_categorical(val_labels, 26)
    
    for i in range(1, 5):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(np.hstack((tr_x_hists, tr_y_hists)), Y_train)
        print(model.score(np.hstack((val_x_hists, val_y_hists)), Y_val))
    #return model.score(np.hstack((val_x_hists, val_y_hists)), Y_val)


"""
Main execution
------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    training_images = image_ops.load_images_npz(input("Enter a file path for the training-npz-data: "))
    validation_images = image_ops.load_images_npz(input("Enter a file path for the validation-npz-data: "))

    '''
    t0 = time.time()

    #for key in images:
    #    print(key, len(images[key]))
    #    images[key] = image_ops.augment_images(images[key], 200, key)
    #    image_ops.save_images_npz("data/Data_Test/Data_" + key, images[key])

    train = {}  # dictionary for the train images
    train.fromkeys(training_images.keys(), [])
    validate_dict = {}  # dictionary for the validation images
    validate_dict.fromkeys(training_images.keys(), [])
    validate = []  # list of imgs used for validation
    validate_labels = []  # list of labels in the same order as 'validate', also used for validation

    count = 0
    for key in training_images:
        image_ops.print_progress_bar(count, 25, prefix='Preparing training-data for {0}'.format(key),
                                     suffix='Complete', length=50)
        for i in range(len(training_images)):
            training_images[key][i] = to_binary(training_images[key][i], 127) * 255   # binarization of all images.
        train[key] = training_images[key]
        count += 1

    count = 0
    for key in validation_images:
        image_ops.print_progress_bar(count, 25, prefix='Preparing validation-data for {0}'.format(key),
                                     suffix='Complete', length=50)
        for i in range(len(validation_images)):
            validation_images[key][i] = to_binary(validation_images[key][i], 127) * 255   # binarization of all images.
        validate_dict[key] = validation_images[key]
        count += 1

    count = 0
    for key in validate_dict:
        validate.append(validate_dict[key])
        for i in range(len(validate_dict[key])):
            validate_labels.append(key)
        count += 1
        image_ops.print_progress_bar(count, 26, prefix='Preparing validation-list for letter {0}'.format(key),
                                     suffix='Complete', length=50)

    print(image_stack(train, validate, validate_labels))

    t1 = time.time()
    print(t1-t0)
    '''
    train = projection_preprocessing(training_images)
    validate = projection_preprocessing(validation_images)
    
    train_projection = []
    train_labels = []
    for key in train:
        for img in train[key]:
            train_projection.append(img)
            train_labels.append(key)
    
    val_projection = []
    val_labels = []
    for key in validate:
        for img in validate[key]:
            val_projection.append(img)
            val_labels.append(key)
    
    print(sklearn_knn(train_projection, val_projection, train_labels, val_labels, 1))
    
    start = time.time()
    guessed_labels = projection(train_projection, val_projection, train_labels)
    end = time.time()
    
    print('Finished in ' + str(round(end - start)) + ' seconds')
    
    count = 0
    for i, guessed_label in enumerate(guessed_labels):
        image_ops.print_progress_bar(i, len(guessed_labels) - 1, 
                                     prefix='Checking predictions',
                                     suffix='Complete', length=50)
        if guessed_label == val_labels[i]: 
            count += 1
    
    print('')
    print('Results of projection: ')
    print(str(count) + '/' + str(len(val_labels)) + ' correct')
    print(str(count / len(guessed_labels) * 100) + '% accuracy')
    
    labeling = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']    
        
    confmat = confusion_matrix(val_labels, guessed_labels, labels=labeling)
    
    confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(26)
    plt.xticks(tick_marks, labeling)
    plt.yticks(tick_marks, labeling)
    fmt = '.2f'
    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        plt.text(j, i, format(confmat[i, j], fmt).lstrip('0'),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if confmat[i, j] > thresh else "black",
                 alpha=0.0 if confmat[i,j] <= 0.1 else 1.0)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

