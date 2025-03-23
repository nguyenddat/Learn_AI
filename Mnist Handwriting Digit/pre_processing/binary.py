import cv2
import numpy as np

def binary(img, threshold = 128):
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh
