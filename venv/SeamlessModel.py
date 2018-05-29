import cv2
import numpy as np

def clone(human, back):
    center = (back.shape[1] // 2, (back.shape[0] // 4)  * 3)
    mask = 255 * np.ones(human.shape, human.dtype)
    return cv2.seamlessClone(human, back, mask, center, cv2.NORMAL_CLONE)