#!/usr/bin/env python
# coding: utf-8

# In[37]:


# importing libraries

import numpy as np
import cv2
import ntpath
from matplotlib import pyplot as plt

# taking image path or name from user

path = input('Enter Image path to apply filter on: ')
input_img = cv2.imread(path)

selections = [int(selection) for selection in input('Select the smoothing filters you want to apply seperated by comma ",": \n1- Blur \n2- Box Filter \n3- Gaussian Blur \n4- Median Blur \n5- Bilateral Filter \n6- Sharpen \n7- Unsharp Masking \nFilters applied:  ').split(",")]

# extracting name of image file from path
name = ntpath.basename(path)
name = name.replace('.png','')

# defining sharpen kernel
sharp = np.array([[-1,-1,-1],
                  [-1,9,-1],
                  [-1,-1,-1]])

# Blur filter
def blur(image):
    kernelSize = (5,5)
    image = cv2.blur(image, kernelSize)
    return image

# Box Filter
def box(image):
    kernelSize = (5,5)
    image = cv2.boxFilter(image,-1,kernelSize)
    return image

# Gaussian Blur filter
def gaussianBlur(image):
    kernelSize = (5,5)
    image = cv2.GaussianBlur(image, kernelSize, cv2.BORDER_DEFAULT)
    return image

# Median filter
def medianBlur(image):
    kernelSize = 3
    image = cv2.medianBlur(image, kernelSize, cv2.BORDER_DEFAULT)
    return image

# Bilateral filter
def bilateral(image):
    dia = 15 # dia of pixel search
    sigmaColor = 70 # color space
    sigmaCoordinate = 75 # coordinate space
    image = cv2.bilateralFilter(image, dia, sigmaColor, sigmaCoordinate)
    return image

# Sharpen
def sharpen(image):
    depth = -1
    image = cv2.filter2D(image, depth, sharp)
    return image

# unsharp masking
def unsharpMask(input_img, output_img):
    alpha = 0.5
    beta = 0.5
    gamma = 0.0
    output_img = cv2.addWeighted(input_img, alpha, output_img, beta, gamma)
    return output_img
    
output_img = input_img

while(True):
    for selection in selections:
        Filters = [blur(output_img), box(output_img), gaussianBlur(output_img), medianBlur(output_img), bilateral(output_img), sharpen(output_img), unsharpMask(input_img, output_img)]
        if selection<7:
            output_img = Filters[selection-1]
            continue
        elif selection == 7:
            output_img = Filters[selection-1]
        else:
            print("invalid selection")
    break

# Rescaling the image
scale_percent = 60 # percent of original size
width = int(output_img.shape[1] * scale_percent /
            100)
height = int(output_img.shape[0] * scale_percent / 100)
dim = (width, height)

output_img = cv2.resize(output_img, dim, interpolation = cv2.INTER_AREA)
input_img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)

while True:
    cv2.imshow('input', input_img)
    cv2.imshow('Output',output_img)
    cv2.imwrite(name+'-improved.png',output_img)
    if cv2.waitKey(0):
        break
cv2.destroyAllWindows()


# In[ ]:




