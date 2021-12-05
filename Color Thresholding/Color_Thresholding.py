#!/usr/bin/env python
# coding: utf-8

# In[14]:


# importing libraries

import cv2
import numpy as np

# Original Image

path = input('Enter the name of image:') # defining path of image (in jupyter it is just name of image)
# taking input from user regarding which regions to focus on 1- Brighter Regions 2- Darker Regions
action = input('To color brighter regions enter 1, for darker regions enter 2: ') 
# using opencv functions to read, display, and save the color image 
color = cv2.imread(path)
cv2.imshow(path, color)
cv2.imwrite(path + '_input.png', color)

# grayscale image (using opencv function to convert color image to grayscale)

grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
path2 = path + '_grayscale.jpg'
cv2.imshow(path2, grayscale)
cv2.imwrite(path2, grayscale)

# binary image with black and white color (using opencv function to convert grayscale image to binary)

path3 = path + '_binary.jpg'
(threshold, binary) = cv2.threshold(grayscale, 95, 255, cv2.THRESH_BINARY)
cv2.imshow(path3, binary)
cv2.imwrite(path3, binary)

# defining number of pixels in the image

(height, width) = binary.shape[:2]


# Binary image with color switching (Switching brighter/darker regions to red from binary image)

# output = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)
# for i in range(height):
#     for j in range(width):
#         if np.array_equal(output[i,j], [255,255,255]) == True:
#             if action == '1':
#                 output[i,j] = (0,0,255)
#             elif action == '2':
#                 output[i,j] = (255,255,255)
#             else:
#                 continue
#         elif np.array_equal(output[i,j], [0,0,0]) == True:
#             if action == '1':
#                 output[i,j] = (0,0,0)
#             elif action == '2':
#                 output[i,j] = (0,0,255)
#             else:
#                 continue
            
#         else:
#             print('invalid')
#             continue

# path4 = path + '_binary_output'
# cv2.imshow(path4, output)
# cv2.waitKey(0)

# Color image with color switching (Switching brighter/darker regions to red from color image)

color2 = color # defining a separate image array to work on
for i in range(height):
    for j in range(width):
        
        if action == '1':                    # for coloring brighter regions red
            A = color2[i,j]>[40,40,40]
            if A.all() == True:
                color2[i,j] = (0,0,255)
            else:
                continue
        
        elif action == '2':                   # for coloring darker regions red
            A = color2[i,j]<[55,55,55]
            if A.all() == True:
                color2[i,j] = (0,0,255)
            else:
                continue
            
        else:                                  # if there's an invalid pixel array
            print('invalid')
            

cv2.imshow(path +'_color_output', color2)
cv2.imwrite(path + '_color_output.jpg', color2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




