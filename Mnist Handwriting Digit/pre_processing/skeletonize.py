import cv2
import numpy as np

def skeletonize(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    eroded = cv2.erode(img, kernel, iterations = 1)

    return eroded