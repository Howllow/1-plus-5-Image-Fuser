import cv2
import numpy as np
import cmath
from WarpModel import warp
from SeamlessModel import clone

def computeAlphaBlending(i1, m1, i2, m2):
    bothmask = cv2.bitwise_or(m1, m2)
    # cv2.namedWindow('bothmask', 0)
    # cv2.imshow('bothmask', bothmask)
    nomask = cv2.bitwise_not(bothmask)
    # cv2.namedWindow('bothmask', 0)
    # cv2.imshow('bothmask', nomask)

    rawAlpha = np.ones((nomask.shape[0], nomask.shape[1]), dtype=float)

    border1 = cv2.convertScaleAbs(255 - border(m1))
    border2 = cv2.convertScaleAbs(255 - border(m2))
    # cv2.namedWindow('bothmask', 0)
    # cv2.imshow('bothmask', border1)

    dist1 = cv2.distanceTransform(border1, cv2.DIST_C, 3)
    d1 = cv2.convertScaleAbs(cv2.threshold(dist1, 0, 255, cv2.THRESH_BINARY)[1])
    # cv2.namedWindow('debug', 0)
    # cv2.imshow('debug', d1)
    tmp_mask = cv2.bitwise_and(m1, d1)
    # cv2.namedWindow('bothmask', 0)
    # cv2.imshow('bothmask', tmp_mask)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist1, tmp_mask)
    dist1 = (dist1 * 1.0) / maxVal

    dist2 = cv2.distanceTransform(border2, cv2.DIST_C, 3)
    d2 = cv2.convertScaleAbs(cv2.threshold(dist2, 0, 255, cv2.THRESH_BINARY)[1])
    # cv2.namedWindow('debug', 0)
    # cv2.imshow('debug', d1)
    tmp_mask = cv2.bitwise_and(m2, d2)
    # cv2.namedWindow('bothmask', 0)
    # cv2.imshow('bothmask', tmp_mask)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist2, tmp_mask)
    dist2 = (dist2 * 1.0)  / maxVal

    dist1Masked = cv2.add(rawAlpha, 0, mask=nomask)
    tmp = cv2.add(dist1, 0, mask=m1)
    dist1Masked += tmp
    dist2Masked = cv2.add(rawAlpha, 0, mask=nomask)
    tmp = cv2.add(dist2, 0, mask=m2)
    dist2Masked += tmp

    dist2Masked = dist2Masked ** 2
    dist1Masked = dist1Masked ** 2
    '''dist1Masked = 1.6 ** np.sqrt(dist1Masked)
    dist2Masked = 1.6 ** np.sqrt(dist2Masked)'''

    blendMaskSum = dist1Masked + dist2Masked

    im1AlphaB, im1AlphaG, im1AlphaR = cv2.split(i1)

    im1AlphaB = im1AlphaB.astype(dist1Masked.dtype) * dist1Masked
    im1AlphaG = im1AlphaG.astype(dist1Masked.dtype) * dist1Masked
    im1AlphaR = im1AlphaR.astype(dist1Masked.dtype) * dist1Masked
    # im1Alpha = cv2.merge([im1AlphaB, im1AlphaG, im1AlphaR])
    # cv2.namedWindow('debug', 0)
    # cv2.imshow('debug', im1Alpha)

    im2AlphaB, im2AlphaG, im2AlphaR = cv2.split(i2)
    im2AlphaB = im2AlphaB.astype(dist1Masked.dtype) * dist2Masked
    im2AlphaG = im2AlphaG.astype(dist1Masked.dtype) * dist2Masked
    im2AlphaR = im2AlphaR.astype(dist1Masked.dtype) * dist2Masked

    imBlendedB = (im1AlphaB + im2AlphaB) / blendMaskSum
    imBlendedG = (im1AlphaG + im2AlphaG) / blendMaskSum
    imBlendedR = (im1AlphaR + im2AlphaR) / blendMaskSum

    imBlend = cv2.merge([imBlendedB, imBlendedG, imBlendedR])
    result = cv2.convertScaleAbs(imBlend)

    return result


def border(mask):
    x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=7)
    y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=7)
    '''
    cv2.namedWindow('x', 0)
    cv2.imshow('x', x)

    cv2.namedWindow('y', 0)
    cv2.imshow('y', y)
    '''
    border = cv2.magnitude(x, y)
    # cv2.namedWindow('border', 0)
    # cv2.imshow('border',border)

    return cv2.threshold(border, 0, 255, cv2.THRESH_BINARY)[1]


img = []

address = "/Users/howllow/Documents/images/"
img.append(cv2.imread(address + 'human2.jpeg'))
img.append(cv2.imread(address + 'up2.jpeg'))
img.append(cv2.imread(address + 'bottomright2.jpeg'))
img.append(cv2.imread(address + 'upright2.jpeg'))
img.append(cv2.imread(address + 'bottomleft2.jpeg'))
img.append(cv2.imread(address + 'upleft2.jpeg'))

'''
img.append(cv2.imread('../photos/2_0.png'))
img.append(cv2.imread('../photos/2_1.png'))
img.append(cv2.imread('../photos/2_2.png'))
'''
cnt = len(img)

img[0] = cv2.copyMakeBorder(img[0], img[0].shape[0], 0, img[0].shape[1] // 2,
                            img[0].shape[1] // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
(row, col) = (img[0].shape[0], img[0].shape[1])

out = img[0]

for i in range(1, cnt):
    img_w = warp(img[i], out)
    img_w[int(0.8 * row):row, int(0.45 * col):int(0.55 * col)] = 0
    # cv2.imshow('out', out)
    mask_origin = cv2.threshold(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    mask_add = cv2.threshold(cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('mask_origin', mask_origin)
    # cv2.imshow('mask_add', mask_add)
    out = computeAlphaBlending(out, mask_origin, img_w, mask_add)
    cv2.imwrite(address+'out1.jpeg', out)
cv2.imshow('aa', out)
cv2.imwrite(address + 'wha.jpeg', out)

# out[int(0.7*row):row, int(0.4*col):int(0.6*col)] = img[0][int(0.7*row):row, int(0.4*col):int(0.6*col)]

# cv2.imwrite('../photos/data/adjust/3_outresult1(2).jpg', out)

cv2.namedWindow('result', 0)
cv2.imshow('result', out)

cv2.waitKey(0)
cv2.destroyAllWindows()