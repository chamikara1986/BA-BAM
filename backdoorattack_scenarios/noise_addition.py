#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""

import os
import cv2
import numpy as np
import glob
epsilon = 0.005
classID = 'George_W_Bush'
if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/clean_noisy_outputs/" + classID + "_" + str(epsilon)):
	os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/clean_noisy_outputs/" + classID + "_" + str(epsilon))

if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_noisy_outputs/" + classID + "_" + str(epsilon)):
    os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_noisy_outputs/" + classID + "_" + str(epsilon))

#method for noise addition
def noiseGen(paths, outputdir, mean_image, maxeqdiff, indexoffset):
    for i, path in enumerate(paths):

        normpath = os.path.normpath(path)
        img = cv2.imread(normpath) #Current image in consideration

        nrsensimg = calEuDistance(mean_image,img)/maxeqdiff #the normalized sensitivity of the current image (the requirement of the noise)

        sensitivity = nrsensimg
        epsilon = 0.005

        laplac = np.random.laplace(0,sensitivity/epsilon,img.size) #Generating laplacian noise based on the sensitivity of the image
        #and the epsilon
        laplac = laplac.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        noise = img + img * laplac #noise addition
        noisypath = outputdir + str(classID) + str(i+1+indexoffset) + ".jpg"
        cv2.imwrite(noisypath, noise)
        print("Original path: ", normpath, ", Output path", noisypath, ", Sens:", nrsensimg)


def calEuDistance(i1,i2): #The method for eucledian distance calculation
    return np.sum((i1-i2)**2)

#classpath1 = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test/Other_350' # clean
classpath2 = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_data_LFW/topaz_08' # poison
outputdir1 = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/clean_noisy_outputs/" + classID + "_" + str(epsilon) + "/"
outputdir2 = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_noisy_outputs/" + classID + "_" + str(epsilon) + "/"

images = [] #Dummy temp array to represent the image list
#paths1 = list(glob.iglob(os.path.join(classpath1, "*.jpg")))
with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/topaz_08/clean_GWB.txt", "r") as f1:
    paths1 = f1.readlines()
    for i in range(len(paths1)):
        paths1[i] = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test/" + paths1[i]
        paths1[i] = paths1[i].rstrip("\n")

paths2 = list(glob.iglob(os.path.join(classpath2, "*.png"))) #poison data path

#add images from first directory to image list
for path in paths1:
    normpath = os.path.normpath(path)
    print("Path1: ", normpath)
    images.append(cv2.imread(normpath))

#add images from first directory to image list
for path in paths2:
    normpath = os.path.normpath(path)
    print("Path2: ", normpath)
    images.append(cv2.imread(normpath))


mean_image = np.mean(([image for image in images]), axis=0) # Get the mean/average image

imdiffmat = [] #list to hold all euclidean distances of all images related to one class
for i in range (0,len(images)):
    imdiff = calEuDistance(mean_image,images[i])
    imdiffmat.append(imdiff)

maxeqdiff = max(imdiffmat) #maximum eucledian distance different of any two images within the class

#First loop for output directory 1
noiseGen(paths1, outputdir1, mean_image, maxeqdiff, 0)

#First loop for output directory 2
noiseGen(paths2, outputdir2, mean_image, maxeqdiff, len(paths1)) #offset index by the length of the previous path list to keep original index values
