#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import ntpath
import math

# Defining a Convolution function
def convolution(img, kernel):
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape
    
    output = np.zeros((img_row-2, img_col-2))
    
    for i in range (img_row-2):
        for j in range (img_col-2):
            output[i,j] = np.sum(kernel * img[i:i+3,j:j+3])
    return output

# Defining Sobel filtering function
def sobel_edge_detection(image):
    
    # defining kernel for gaussian blur
    kernelSize = (3,3)
    image = cv2.GaussianBlur(image, kernelSize, cv2.BORDER_DEFAULT)

    # Defining sobel filter kernel
    filter = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])
    
    # horizontal sobel
    sobel_h = convolution(image, filter)
    
    # vertical sobel
    sobel_v = convolution(image, np.flip(filter.T, axis=0))
    
    # combining
    sobel = np.sqrt(np.square(sobel_h) + np.square(sobel_v))
    
    # Normalizing
    sobel *= 255.0 / (np.max(sobel))
    
    #Inverting colors
    (thresh, sobel) = cv2.threshold(sobel, 15, 255, cv2.THRESH_BINARY_INV)
    return sobel

# Taking image path or name from user

path = input('Enter Image path to apply sobel edge detection on: ')
input_img = cv2.imread(path,0)

# Extracting name of image file from path
name = ntpath.basename(path)
name = name.replace('.png','')

# Applying sobel filter
output_img = sobel_edge_detection(input_img)

# Rescaling the image
scale_percent = 60 # percent of original size
width = int(output_img.shape[1] * scale_percent / 100)
height = int(output_img.shape[0] * scale_percent / 100)
dim = (width, height)
output_img = cv2.resize(output_img, dim, interpolation = cv2.INTER_AREA)
input_img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("Sobel_out", output_img)
cv2.imshow("Input", input_img)

# Canny Edge detection function
def can(threshold1=0):
    threshold1 = cv2.getTrackbarPos('threshold1', 'Canny_Out')
    threshold2 = cv2.getTrackbarPos('threshold2', 'Canny_Out')
    apertureSize = int(cv2.getTrackbarPos('apertureSize', 'Canny_Out'))
    if ((apertureSize > 3 and apertureSize < 5) or apertureSize == 3) :
        apertureSize = 3
    if ((apertureSize > 5 and apertureSize < 7 )or apertureSize == 5) :
        apertureSize = 5
    L2gradient = cv2.getTrackbarPos('L2gradient', 'Canny_Out')
    if L2gradient == 0:
        L2gradient = True
    else:
        L2gradient = False
    edge = cv2.Canny(img, threshold1, threshold2, apertureSize = apertureSize, L2gradient = L2gradient)
    edge = 255 - edge
    cv2.resize(edge, (1000, 800))
    cv2.imshow('Canny_Out', edge)
    cv2.imwrite(name+'-canny.png',edge)

# creating trackbars
img = input_img.copy()
img = cv2.GaussianBlur(img,(5,5),0)
cv2.namedWindow('Canny_Out', cv2.WINDOW_NORMAL)
threshold1=100
threshold2=1
cv2.createTrackbar('threshold1','Canny_Out',threshold1,255,can)
cv2.createTrackbar('threshold2','Canny_Out',threshold2,255,can)
cv2.createTrackbar('apertureSize','Canny_Out',3,7,can)
cv2.createTrackbar('L2gradient','Canny_Out',0,1,can)
can(0)

# Displaying Image
cv2.imwrite(name+'-sobel.png',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




