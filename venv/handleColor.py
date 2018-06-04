from PIL import Image
from PIL import ImageSequence
from PIL import ImageStat
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import heapq

# 得到目标图像里面的色温信息
def get_color_temp(new_pixels):
    allSize = len(new_pixels)
    if allSize == 0:
        return [1.0,1.0,1.0]
    parameters = []
    for color in range(0, 3):
        sum = 0
        for pos in range(0, allSize):
            sum += new_pixels[pos][color]
        parameters.append(float(sum) / float(allSize))
    print("参数",parameters)
    para = (parameters[0] + 2 * parameters[1] + parameters[2]) / 4
    paras = [para / parameters[0], para / parameters[1], para / parameters[2]]
    # for i in range(0,3):
    #    paras[i]=1 + 500*(paras[i]-1.0)
    # print(paras)
    return paras


# 改变原图像的色温，使之接近目标图像
def set_color_temp(pixels, para):
    allSize = len(pixels)
    for pos in range(0, allSize):
        r_pixel, g_pixel, b_pixel = pixels[pos][0], pixels[pos][1], pixels[pos][2]
        new_pixel = (int(r_pixel / r_para), int(g_pixel / g_para), int(b_pixel / b_para))
        new_pixels.append(new_pixel)
    return new_pixels


# 改变自己的色温，使得每个颜色的权重相同
def repair_color_temp(im_open_file, parameters):
    pixels = list(im_open_file.getdata())
    allSize = len(pixels)
    r_para, g_para, b_para = parameters[0], parameters[1], parameters[2]
    new_pixels = []
    for pos in range(0, allSize):
        r_pixel, g_pixel, b_pixel = pixels[pos][0], pixels[pos][1], pixels[pos][2]
        new_pixel = (int(r_pixel / r_para), int(g_pixel / g_para), int(b_pixel / b_para))
        new_pixels.append(new_pixel)
    return new_pixels


# 转换颜色空间，将RGB空间转换为YCbCr空间
def tranfer_space(pixel):
    Y = 0.2990 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2]
    Cb = -0.1687 * pixel[0] - 0.3313 * pixel[1] + 0.5000 * pixel[2]
    Cr = 0.5000 * pixel[0] - 0.4187 * pixel[1] - 0.0813 * pixel[2]
    return [Y, Cb, Cr]


# 将YCbCr空间转换回RGB空间
def retranfer_space(new_pixel):
    R = new_pixel[0] + 1.4020 * new_pixel[2]
    G = new_pixel[0] - 0.3441 * new_pixel[1] - 0.7141 * new_pixel[2]
    B = new_pixel[0] + 1.7720 * new_pixel[1] - 0.0001 * new_pixel[2]
    return [int(R + 0.2), int(G + 0.2), int(B + 0.2)]


# 使用边界条件筛选出相关的白点
def filter_white(im_open_file):
    pixels = list(im_open_file.getdata())
    new_pixels = list(map(lambda x: tranfer_space(x), pixels))
    allSize = len(new_pixels)
    new_pixels_Y = list(map(lambda x: x[0], new_pixels))
    new_pixels_Cb = list(map(lambda x: math.fabs(x[1]), new_pixels))
    new_pixels_Cr = list(map(lambda x: math.fabs(x[2]), new_pixels))
    new_pixels_Y.sort()
    new_pixels_Cb.sort()
    new_pixels_Cr.sort()
    Y_bound = new_pixels_Y[int(allSize * 0.9)]
    Cb_bound = new_pixels_Cb[int(allSize * 0.1)]
    Cr_bound = new_pixels_Cr[int(allSize * 0.1)]
    new_filter_pixels = \
        list(filter(lambda x: filter_fun(Y_bound, Cb_bound, Cr_bound, x), new_pixels))
    filter_pixels = list(map(lambda x: retranfer_space(x), new_filter_pixels))
    filter_pixels2 = list(filter(lambda x: filter_fun2(x), filter_pixels))
    return filter_pixels2


def filter_fun(Y_bound, Cb_bound, Cr_bound, new_pixel):
    return new_pixel[0] > Y_bound \
           and math.fabs(new_pixel[1]) < Cb_bound \
           and math.fabs(new_pixel[2]) < Cr_bound


def filter_fun2(pixel):
    return not (pixel[0] == pixel[1] and pixel[1] == pixel[2])


# 整个色温转换的过程
def coltmp_tranfer(source_im):
    source_filter_pixels = filter_white(source_im)
    # source_filter_pixels = filter_white(source_im)
    source_para = get_color_temp(source_filter_pixels)
    # source_para = get_color_temp(source_filter_pixels)
    source_repair_pixels = repair_color_temp(source_im, source_para)
    mode = source_im.mode
    width, height = source_im.size
    new_source_im = Image.new(mode, (width, height))
    new_source_im.putdata(data=source_repair_pixels)
    return new_source_im


