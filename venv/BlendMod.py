import cv2
import numpy as np
import cmath
from WarpMod import warp
from FaceDectMod import FaceDetect


def computeAlphaBlending(i1, m1, i2, m2):
    bothmask = cv2.bitwise_or(m1, m2)
    nomask = cv2.bitwise_not(bothmask)

    rawAlpha = np.ones((nomask.shape[0], nomask.shape[1]), dtype=float)

    border1 = cv2.convertScaleAbs(255 - border(m1))
    border2 = cv2.convertScaleAbs(255 - border(m2))

    dist1 = cv2.distanceTransform(border1, cv2.DIST_C, 3)
    d1 = cv2.convertScaleAbs(cv2.threshold(dist1, 0, 255, cv2.THRESH_BINARY)[1])
    tmp_mask = cv2.bitwise_and(m1, d1)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist1, tmp_mask)
    dist1 = (dist1 * 1.0) / maxVal

    dist2 = cv2.distanceTransform(border2, cv2.DIST_C, 3)
    d2 = cv2.convertScaleAbs(cv2.threshold(dist2, 0, 255, cv2.THRESH_BINARY)[1])
    tmp_mask = cv2.bitwise_and(m2, d2)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist2, tmp_mask)
    dist2 = (dist2 * 1.0) / maxVal

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

    border = cv2.magnitude(x, y)

    return cv2.threshold(border, 0, 255, cv2.THRESH_BINARY)[1]


def blendProcess(img):
    address = "./image/"
    cnt = len(img)
    for i in range(0, cnt):
        bias = np.ones(img[i].shape, dtype=np.uint8)
        img[i] = cv2.add(img[i], bias)
    img[0] = cv2.copyMakeBorder(img[0], img[0].shape[0], 0, img[0].shape[1] // 2,
                                img[0].shape[1] // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    (row, col) = (img[0].shape[0], img[0].shape[1])
    out = img[0]
    faces = FaceDetect(img[0])
    hx, hy, hw, hh = faces
    '''
    tst = img[0]
    cv2.rectangle(tst, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 2)
    cv2.imshow('face', tst)
    cv2.imwrite(address + 'facetest.jpeg', tst)
    '''
    for i in range(1, cnt):
        img_w = warp(img[i], out)
        img_w[hy - hh // 4: row, hx - int(hw * 1.5): hx + int(hw * 2.5)] = 0
        mask_origin = cv2.threshold(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        mask_add = cv2.threshold(cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        out = computeAlphaBlending(out, mask_origin, img_w, mask_add)

    cv2.imwrite(address + 'result.jpg', out)

    cv2.namedWindow('result', 0)
    cv2.imshow('result', out)
    cv2.destroyAllWindows()
