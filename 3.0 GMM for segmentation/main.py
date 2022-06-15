# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

# reading of images
img = cv2.imread('plant_cell.jpg')

# reshaping of above image shape
img2 = img.reshape((-1,3))

cv2.imshow("Frame",img)



# waitkey, delays the image opening time, but 0 means forever
cv2.waitKey(0)
cv2.destroyAllWindows()
