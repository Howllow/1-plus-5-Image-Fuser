from PIL import Image
from PIL import ImageSequence
from PIL import ImageStat
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import heapq


# 使用rms和感知亮度公式计算得到一个照片的亮度
def bright_ness(im_open_file):
    stat = ImageStat.Stat(im_open_file)
    r, g, b = stat.rms
    return 0.2990 * r + 0.5870 * g + 0.1140 * b
    # return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


# 计算mask情况下的亮度
def weighted_bright_ness(image, mask):
    width, height = image.size
    image_pixels = list(image.getdata())
    # print(image_pixels[0],"haha")
    mask_pixels = list(mask.getdata())
    # print(mask_pixels[20000],"sb")
    allSize = width * height
    sumR, sumG, sumB = 0, 0, 0
    # print(mask_pixels)
    for pos in range(0, allSize):
        mask_pixel = mask_pixels[pos]
        if mask_pixel <= 0:
            continue
        image_pixel = image_pixels[pos]
        sumR += image_pixel[0]
        sumG += image_pixel[1]
        sumB += image_pixel[2]
    aveR = float(sumR) / allSize
    aveG = float(sumG) / allSize
    aveB = float(sumB) / allSize
    return 0.2990 * aveR + 0.5870 * aveG + 0.1140 * aveB


# 调整源照片的亮度使之尽量和目标照片相似
def bright_tranfer(source_im, source_bright, target_bright):
    rate_value = target_bright / source_bright
    # print(rate_value)
    pixels = list(source_im.getdata())
    # print(pixels[0:10])
    new_pixels = []
    gamma_map = get_gamma(rate_value)
    for p in pixels:
        p = pixel_bright_tranfer(p, gamma_map)
        new_pixels.append(p)
    # print(new_pixels[0:10])
    mode = source_im.mode
    width, height = source_im.size
    new_im = Image.new(mode, (width, height))
    new_im.putdata(data=new_pixels)
    # print(bright_ness(source_im),bright_ness(new_im),bright_ness(target_im))
    return new_im


# 在有mask的前提下进行光照重整
def weighted_bright_tranfer(human, scene, mask, lap):
    # print(list(mask.getdata()))
    human_bright = weighted_bright_ness(human, mask)
    lap_bright = weighted_bright_ness(lap, mask)
    new_scene = bright_tranfer(scene, lap_bright, human_bright)
    return new_scene


# 调整像素点的亮度
def pixel_bright_tranfer(source_pixel, gamma_map):
    # print(rate_value)
    res_r = res_g = res_b = 0;
    res_r = source_pixel[0]
    res_g = source_pixel[1]
    res_b = source_pixel[2]
    res_r = min(255, gamma_map[res_r])
    res_g = min(255, gamma_map[res_g])
    res_b = min(255, gamma_map[res_b])
    return (int(res_r), int(res_g), int(res_b))


# 为了实现加速，建立gamma函数映射表
def get_gamma(rate_value):
    gamma_res = []
    if rate_value > 1:
        rate_value = 1 + (rate_value - 1) * 1
    elif rate_value < 1:
        rate_value = 1 - (1 - rate_value) * 1
    else:
        rate_value = 1
    for i in range(0, 256):
        now_gamma = int((math.pow((i + 0.5) / 256, 1 / rate_value)) * 256 - 0.5)
        # math.exp(math.log((i + 0.5) / 256.0 * rate_value)) * 255.0
        gamma_res.append(now_gamma)
    return gamma_res
