import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

address = "/Users/howllow/Desktop/"
human = cv2.imread(address + "human3.jpeg")
back = cv2.imread(address + "backall.jpeg")
center = (back.shape[1]//2, (back.shape[0]//4) * 3)
mask = 255 * np.ones(human.shape, human.dtype)

clone = cv2.seamlessClone(human, back, mask, center, cv2.NORMAL_CLONE)

cv2.imshow('res', clone)
cv2.imwrite(address + "seamless.jpeg", clone)
