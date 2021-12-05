#!/usr/bin/env python
# coding: utf-8

# In[19]:


# importing libraries

import numpy as np
import cv2
import ntpath

# taking image path or name from user

path = input('Enter Image path to convert it to a pseudo colored image: ')
image_color = cv2.imread(path)
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# extracting name of image file from path
name = ntpath.basename(path)
name = name.replace('.png','')


grayscale = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

# Converting to HSV image

image_HSV = cv2.cvtColor(image_color, cv2.COLOR_RGB2HSV)

# Creating Lookup Table

H_lut = np.ones((np.shape(image_HSV[:,:,0])),np.uint8)
S_lut = np.ones((np.shape(image_HSV[:,:,0])),np.uint8)*255
V_lut = np.ones((np.shape(image_HSV[:,:,0])),np.uint8)*255

# calculating and matching lookup table for Hue values

H_lut = np.uint8(140*((grayscale-np.min(grayscale))/(np.max(grayscale)-np.min(grayscale))))

# making output HSV image

output_HSV = np.stack((H_lut, S_lut, V_lut), axis=-1)

# Converting HSV to RGB for display 

output_color = cv2.cvtColor(output_HSV, cv2.COLOR_HSV2RGB)

# finding max brightness spot

Max = np.where(grayscale == np.max(grayscale)) # array with brightest pixel indices
Max = np.transpose(np.array(Max))

a,b = [],[]                                    # arrays for storing adjecent pixel indices

if (len(Max)>1):                               # searching for the center of gravity
    for i in range(len(Max)-1):  
        if (Max[i,0]+1 == Max[i+1,0] or Max[i,0] == Max[i+1,0]):
            a.append(Max[i,0])
            b.append(Max[i,1])
        else:
            continue
else:
    a.append(Max[:,0])
    b.append(Max[:,1])


# indices of brightest spot

max_j = int(a[int(len(a)/2)])
max_i = int(b[int(len(b)/2)])


# drawing circle & marker
radius = 20
color = (255,255,255)
thickness = 2
center = (max_i, max_j)
marker_size = 80
output_color = cv2.circle(output_color, center, radius, color, thickness) 
output_color = cv2.drawMarker (output_color, center, color, 0, marker_size, thickness=1) 

# displaying original image
cv2.imshow('original', image_color) 

 # displaying pseudo colored image
cv2.imshow('Pseudo Colored', output_color)

# saving the image

cv2.imwrite(name +'-color.png',output_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
                


# In[14]:


Max


# In[ ]:




