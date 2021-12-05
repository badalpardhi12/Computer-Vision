#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing libraries

import numpy as np
import cv2

# defining a function for gamma correction
def GammaC(src, gamma):
    invGamma = 1 / gamma # taking inverse gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]   # lookup table calculation
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table) # returning gamma corrected value

# taking image path or name from user

path = input('Enter the image you want to perform gamma correction on: ')
image = cv2.imread(path)

# setting up flag variable
done = 1
# repeatative loop for iterating on value of gamma
while (done==1):
    gamma = float(input('Enter the gamma correction value: ')) # taking input gamma value
    image_g = GammaC(image, gamma) # calling gamma function

    cv2.imshow('original', image) # displaying original image
    cv2.imshow('gamma corrected', image_g) # displaying gamma corrected image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # asking user if the value of gamma needs to change or save the current configuration
    done = int(input('Enter 1 if you want to change gamma value, enter 2 if you want to save the corrected image:'))

if done == 2:
    cv2.imwrite(path+'_corrected.jpg', image_g)
        


# In[ ]:





# In[ ]:


print

