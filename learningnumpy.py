'''
This is the intro into learning cv, starting with numpy.
'''

import cv2 # import all libraries necessary
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("banana.jpg")        # BGR image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # change from bgr to rbg for cv

print(img.shape)     # (height, width, 3)
print(img.dtype)     # uint8

'''
now lets split channels
'''

r = img[:, :, 0] # red light
g = img[:, :, 1] # green light
b = img[:, :, 2] # blue light

plt.imshow (b, cmap = "gray")
plt.title("red channel")
plt.show()


 # this converts the image to gray which is good for removing complexity and preserving shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray, cmap = "gray")
plt.show()


