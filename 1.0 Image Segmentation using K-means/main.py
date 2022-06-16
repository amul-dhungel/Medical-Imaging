import numpy as np 
import cv2 

img = cv2.imread('BSE_Image.jpg')

# we need to reshape the image into 1 d array
img2 = img.reshape((-1,3))

# k-means clustering only accepts float32
img2 = np.float32(img2)

# this tells when max accuracy is reached or max iteration
# is reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# clusters
k = 4
attempts = 10

# using of k-means algorithm from open-cv
ret,label,center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

# convert the above float into unsigned-integer-8
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

# saving it
cv2.imwrite('segmented.jpg',res2)