#Library
from scipy import misc
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Opening image
im = cv2.imread('im4-c.bmp',0)
misc.imsave('face.png', im) # uses the Image module (PIL)
#im.show()

#identifing edges
edges = cv2.Canny(im,100,200)

#extrapulates data from image into array
pixeldata = misc.imread("im1-c.bmp")

#sets up plots in matlabplot
plt.subplot(121),plt.imshow(im,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


#shows plot
plt.show()

