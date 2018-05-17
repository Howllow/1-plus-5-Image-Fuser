import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread("/Users/howllow/Desktop/bottomleft2.jpeg")
img2 = cv2.imread("/Users/howllow/Desktop/up2.jpeg")
human = cv2.imread("/Users/howllow/Desktop/human2.jpeg")
human = cv2.copyMakeBorder(human, human.shape[0], 0, human.shape[1]//2,
                          human.shape[1]//2, cv2.BORDER_CONSTANT,value = [0, 0, 0])
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
grayh = cv2.cvtColor(human, cv2.COLOR_BGR2GRAY)


if img1 is None or img2 is None or human is None :
    print("Reading Image Failed")
    quit()

#cv2.imshow("image1", img1)
#cv2.imshow("image2", img2)
#cv2.imshow("human", human)

#get sift feature
Sift = cv2.xfeatures2d.SIFT_create(800)

kp1, des1 = Sift.detectAndCompute(gray1, None)
kp2, des2 = Sift.detectAndCompute(gray2, None)
kph, desh = Sift.detectAndCompute(grayh, None)

#match with brute-force
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(des1, desh, k = 2) #k = 2 in order to use ratio test
matches2 = bf.knnMatch(des2, desh, k = 2)

#ratio test
good1 = []
good2 = []
for m, n in matches1:
    if m.distance < 0.8 * n.distance:
        good1.append(m)

for m, n in matches2:
    if m.distance < 0.8 * n.distance:
        good2.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
dst_pts = np.float32([ kph[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask1 = mask.ravel().tolist()
change1 = cv2.warpPerspective(img1, M, (human.shape[1], human.shape[0]) )
cv2.imshow('change1', change1)

src_pts = np.float32([ kp2[m.queryIdx].pt for m in good2 ]).reshape(-1,1,2)
dst_pts = np.float32([ kph[m.trainIdx].pt for m in good2 ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask2 = mask.ravel().tolist()
change2 = cv2.warpPerspective(img2, M, (human.shape[1], human.shape[0]) )
cv2.imshow('change2', change2)

#draw
draw_params1 = dict(matchesMask = matchesMask1, flags = 2)
draw_params2 = dict(matchesMask = matchesMask2, flags = 2)
m1 = cv2.drawMatches(img1, kp1, human, kph, good1, None, **draw_params1)
cv2.imshow('test1', m1)
m2 = cv2.drawMatches(img2, kp2, human, kph, good2, None, **draw_params2)
cv2.imshow('test2', m2)

cv2.imwrite('/Users/howllow/Desktop/out1.jpg', change1)
cv2.imwrite('/Users/howllow/Desktop/out2.jpg', change2)
cv2.imwrite('/Users/howllow/Desktop/person.jpg', human)
res1 = Image.open('/Users/howllow/Desktop/out1.jpg')
res2 = Image.open('/Users/howllow/Desktop/out2.jpg')
person = Image.open('/Users/howllow/Desktop/person.jpg')

'''person1 = Image.blend(person, res1, 0.7)
person2 = Image.blend(person, res2, 0.7)
person = Image.blend(person1, person2, 0.5)
person.show()
person.save('/Users/howllow/Desktop/final.png')'''
m = 0
n = 0
print(human.shape[0])
for m in range(human.shape[0]) :
    for n in range(human.shape[1]) :
        if  (change1[m][n]!=[0,0,0]).all() and (human[m][n]==[0,0,0]).all():
            human[m][n] = change1[m][n]
        if  (change2[m][n]!=[0,0,0]).all() and (human[m][n]==[0,0,0]).all():
            human[m][n] = change2[m][n]




cv2.imshow('res', human)
cv2.imwrite('/Users/howllow/Desktop/final.png', human)

cv2.waitKey(0)
