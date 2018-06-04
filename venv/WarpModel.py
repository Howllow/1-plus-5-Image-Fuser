import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def warp (srcImage, dstImage):

    #change to gray
    gray_src = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(dstImage, cv2.COLOR_BGR2GRAY)

    # get sift feature
    SIFT = cv2.xfeatures2d.SIFT_create()
    kp_src, des_src = SIFT.detectAndCompute(gray_src, None)
    kp_dst, des_dst = SIFT.detectAndCompute(gray_dst, None)

    # match with brute-force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_src, des_dst, k = 2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.73 * n.distance:
            good.append(m)

    # Get warped images
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
    return cv2.warpPerspective(srcImage, M, (dstImage.shape[1], dstImage.shape[0]))












