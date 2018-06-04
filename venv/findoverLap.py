import cv2
import numpy as np
from WarpModel import warp

def findOverlap(human, scene):

    img1 = human
    img2 = scene

    bias = np.ones(img1.shape, dtype = np.uint8)
    img1 = cv2.add(img1, bias)

    bias = np.ones(img2.shape, dtype = np.uint8)
    img2 = cv2.add(img2, bias)

    img_w = warp(img2, img1)

    mask = cv2.threshold(cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]

    #cv2.imshow('out',img_w)
    #cv2.imshow('origin', img1)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img_w, mask