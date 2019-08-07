#import the necessary packages 
import numpy as np 
import cv2 
from scipy import misc

#sets the images to variables
image1 = 'im1-c.bmp'
image2 = 'im2-c.bmp'
image3 = 'im3-c.bmp'
image4 = 'im4-c.bmp'
image5 = 'im5-c.bmp'
image6 = 'im6-c.bmp'
image7 = 'im7-c.bmp'
lowerbound = 0
upperbound = 0
#sets the current test image to be used
test_image = image7

image = cv2.imread(test_image)
#sets the contour bounds for each image 
if test_image == image1:
    lowerbound = 2000
    upperbound = 10000
elif test_image == image2:
    lowerbound = 1000
    upperbound = 10000
elif test_image == image3:
    lowerbound = 5000
    upperbound = 6000
elif test_image == image4:
    lowerbound = 1000
    upperbound = 4000
elif test_image == image5:
    lowerbound = 1000
    upperbound = 5000
elif test_image == image6:
    lowerbound = 1000
    upperbound = 5000
elif test_image == image7:
    lowerbound = 700
    upperbound = 900

#creates the kernel that is used to pass across the image
kernel = np.ones((30,30), np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#processing
gray = cv2.erode(gray, kernel, iterations = 2)
gray = cv2.dilate(gray, kernel, iterations = 3)

gray = cv2.bilateralFilter(gray, 30,30,30)

cv2.imshow("Gray", gray)
cv2.waitKey(0)

#detect edges in the image 
edged = cv2.Canny (gray, 10, 250)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

#construct and apply a closing kernel to 'close' gaps between 'white' pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0) 

#find contours (i.e the outlines) in the image and initialize the total number of books found
_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
#loop over the contours 
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.01 * peri, True)

    if peri > lowerbound and peri < upperbound:
        cv2.drawContours(image, [approx], -1, (255,255,0), 3)
        total += 1

cv2.imshow("Output", image)
cv2.waitKey(0)

misc.imsave("face.png", image)