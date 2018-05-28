import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#input images
address = "/Users/howllow/Documents/images/"
upleft = cv2.imread(address + "upleft4.jpeg")
bottomleft = cv2.imread(address + "bottomleft4.jpeg")
upright = cv2.imread(address + "upright4.jpeg")
bottomright = cv2.imread(address + "bottomright4.jpeg")
up = cv2.imread(address + "up4.jpeg")
human = cv2.imread(address + "human4.jpeg")
orihuman = human
human = cv2.copyMakeBorder(human, human.shape[0], 0, human.shape[1]//2,
                          human.shape[1]//2, cv2.BORDER_CONSTANT,value = [0, 0, 0])


#change to gray
grayul = cv2.cvtColor(upleft, cv2.COLOR_BGR2GRAY)
graybl = cv2.cvtColor(bottomleft, cv2.COLOR_BGR2GRAY)
grayur = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY)
graybr = cv2.cvtColor(bottomright, cv2.COLOR_BGR2GRAY)
grayu  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
grayh = cv2.cvtColor(human, cv2.COLOR_BGR2GRAY)


#get sift feature
Sift = cv2.xfeatures2d.SIFT_create()
kpul, desul = Sift.detectAndCompute(grayul, None)
kpbl, desbl = Sift.detectAndCompute(graybl, None)
kpur, desur = Sift.detectAndCompute(grayur, None)
kpbr, desbr = Sift.detectAndCompute(graybr, None)
kpu, desu = Sift.detectAndCompute(grayu, None)
cv2.imwrite(address + "uu.jpeg",grayu)
kph, desh = Sift.detectAndCompute(grayh, None)


#match with brute-force
bf = cv2.BFMatcher()
matchesul = bf.knnMatch(desul, desh, k = 2)
matchesbl = bf.knnMatch(desbl, desh, k = 2)
matchesur = bf.knnMatch(desur, desh, k = 2)
matchesbr = bf.knnMatch(desbr, desh, k = 2)
matchesu = bf.knnMatch(desu, desh, k = 2)


#ratio test
goodul = []
goodbl = []
goodur = []
goodbr = []
goodu = []


def ratiotest(matches, good):
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)


ratiotest(matchesul, goodul)
ratiotest(matchesbl, goodbl)
ratiotest(matchesur, goodur)
ratiotest(matchesbr, goodbr)
ratiotest(matchesu, goodu)


#Get changed images
def Warp(good, kp, image):
    src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kph[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return mask.ravel().tolist(), cv2.warpPerspective(image, M, (human.shape[1], human.shape[0]))


matchesMaskul, changeul = Warp(goodul, kpul, upleft)
cv2.imwrite(address + 'c_ul.jpeg', changeul)
matchesMaskbl, changebl = Warp(goodbl, kpbl, bottomleft)
cv2.imwrite(address + 'c_bl.jpeg', changebl)
matchesMaskur, changeur = Warp(goodur, kpur, upright)
cv2.imwrite(address + 'c_ur.jpeg', changeur)
matchesMaskbr, changebr = Warp(goodbr, kpbr, bottomright)
cv2.imwrite(address + 'c_br.jpeg', changebr)
matchesMasku, changeu = Warp(goodu, kpu, up)
cv2.imwrite(address + 'c_u.jpeg', changeu)


#draw
draw_paramsul = dict(matchesMask = matchesMaskul, flags = 2)
draw_paramsbl = dict(matchesMask = matchesMaskbl, flags = 2)
draw_paramsur = dict(matchesMask = matchesMaskur, flags = 2)
draw_paramsbr = dict(matchesMask = matchesMaskbr, flags = 2)
draw_paramsu  = dict(matchesMask = matchesMasku,  flags = 2)
m_ul = cv2.drawMatches(upleft, kpul, human, kph, goodul, None, **draw_paramsul)
m_bl = cv2.drawMatches(bottomleft, kpbl, human, kph, goodbl, None, **draw_paramsbl)
m_ur = cv2.drawMatches(upright, kpur, human, kph, goodur, None, **draw_paramsur)
m_br = cv2.drawMatches(bottomright, kpbr, human, kph, goodbr, None, **draw_paramsbr)
m_u = cv2.drawMatches(up, kpu, human, kph, goodu, None, **draw_paramsu)


cv2.imshow('upleftmatch', m_ul)
cv2.imwrite(address + 'm_ul.jpeg', m_ul)
cv2.imshow('bottomleftmatch', m_bl)
cv2.imwrite(address + 'm_bl.jpeg', m_bl)
cv2.imshow('uprightmatch', m_ur)
cv2.imwrite(address + 'm_ur.jpeg', m_ur)
cv2.imshow('bottomrightmatch', m_br)
cv2.imwrite(address + 'm_br.jpeg', m_br)
cv2.imshow('upmatch', m_u)
cv2.imwrite(address + 'm_u.jpeg', m_u)


#stitch!
def stitch2(image1, image2):
    back = image1
    for m in range(human.shape[0]) :
        for n in range(human.shape[1]) :
            if (image2[m][n] != [0, 0, 0]).any() and (image1[m][n] == [0, 0, 0]).all():
                back[m][n] = image2[m][n]
            if (image1[m][n] != [0, 0, 0]).all() and (image2[m][n] != [0, 0, 0]).all():
                if (image2[m][n] > image1[m][n]).all():
                    back[m][n] = image2[m][n]
    return back


#get background
back_up = stitch2(changeul, changeur)
cv2.imshow('backup', back_up)
cv2.imwrite(address + "backup.jpeg", back_up)
back_bottom = stitch2(changebl, changebr)
cv2.imshow('backb', back_bottom)
cv2.imwrite(address + "backb.jpeg", back_bottom)
back_ub = stitch2(back_bottom, back_up)
cv2.imshow('backub', back_ub)
cv2.imwrite(address + "backub.jpeg", back_ub)
back_all = stitch2(back_ub, changeu)
cv2.imshow('back_all', back_all)
cv2.imwrite(address + "backall.jpeg", back_all)


#clone human to background
center = (human.shape[1] // 2, (human.shape[0] // 4) * 3)
mask = 255 * np.ones(orihuman.shape, orihuman.dtype)
res = cv2.seamlessClone(orihuman, back_all, mask, center, cv2.NORMAL_CLONE)


cv2.imwrite(address + 'final.jpeg', res)


cv2.waitKey(0)










