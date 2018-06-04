import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

address = "/Users/howllow/Documents/images/"
img1 = cv2.imread(address + 'human2.jpeg')
img2 = cv2.imread(address + 'bottomleft2.jpeg')
# change to gray
gray_src = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_dst = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('img1', img1)
# get sift feature
SIFT = cv2.xfeatures2d.SIFT_create()
#SIFT = cv2.xfeatures2d.SURT_create()
#SIFT = cv2.ORB_create()
kp_src, des_src = SIFT.detectAndCompute(gray_src, None)
kp_dst, des_dst = SIFT.detectAndCompute(gray_dst, None)

# match with brute-force
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_src, des_dst, k=2)

#match with flann based
# FLANN parameters
'''FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des_src,des_dst,k=2)'''

# ratio test
good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

# Get warped images
src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)[0]
# cv2.drawMatchesKnn expects list of lists as matches.
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp_src,img2,kp_dst,good,None,**draw_params)
cv2.imshow('img3', img3)
#cv2.imwrite(address+"orbandbf.jpeg",img3)
cv2.imwrite(address+"siftandbf.jpeg",img3)
#cv2.imwrite(address+"surfandfl.jpeg",img3)
