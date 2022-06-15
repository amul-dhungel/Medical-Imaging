from skimage import io, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma


# read the images
img = img_as_float(io.imread("Alloy_noisy.jpg"))



# denoise the image to imporve the historgram
# to denoise it , skimage non-local means package is used
sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))


patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)

# slow algorithm of non-local means
# fast mode (True) is overdoing 
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)


