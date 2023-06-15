import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.filters import fspecial, filter2D

def similarity(path1, path2):

    # Load the two images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the FSIM between the two images
    kernel = fspecial('gaussian', 11, 1.5)
    mu1 = filter2D(gray1, kernel)
    mu2 = filter2D(gray2, kernel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2D(gray1 ** 2, kernel) - mu1_sq
    sigma2_sq = filter2D(gray2 ** 2, kernel) - mu2_sq
    sigma12 = filter2D(gray1 * gray2, kernel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + 0.0001) * (2 * sigma12 + 0.0009)) / ((mu1_sq + mu2_sq + 0.0001) * (sigma1_sq + sigma2_sq + 0.0009))
    fsim = np.mean(ssim_map)

    return fsim
