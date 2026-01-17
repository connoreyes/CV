import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("house.jpg") # read the image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image from BGR to GRAY (grayscale)


'''
# split channels
r = img[:, :, 0] # red light
g = img[:, :, 1] # green light
b = img[:, :, 2] # blue light
'''
# Sobel X: detects vertical edges (left↔right intensity change)
sobel_x = np.array([ 
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

# Sobel Y: detects horizontal edges (up↕down intensity change)
sobel_y = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

height, width = gray.shape

gx = np.zeros((height, width)) # this will store sobel_x results
gy = np.zeros((height, width)) # this will store sobel_y results

for y in range(1, height -1): # both loops ignore the borders of the pixels 
    for x in range(1, width -1):
        region = gray[y-1:y+2, x-1:x+2] # this creates the region of 3x3 
        gx[y,x] = np.sum(region * sobel_x) # convolution with Sobel X
        gy[y,x] = np.sum(region * sobel_y) # # convolution with Sobel Y 

magnitude = np.sqrt(gx**2 + gy**2) # combine the total edge strength

magnitude = magnitude / magnitude.max() * 255 # scale the value

magnitude = magnitude.astype(np.uint8)

plt.figure(figsize=(12,4))


plt.subplot(1,3,1)
plt.title("Sobel X")
plt.imshow(gx, cmap="gray")

plt.subplot(1,3,2)
plt.title("Sobel Y")
plt.imshow(gy, cmap="gray")

plt.subplot(1,3,3)
plt.title("Edge Magnitude")
plt.imshow(magnitude, cmap="gray")

plt.show()
