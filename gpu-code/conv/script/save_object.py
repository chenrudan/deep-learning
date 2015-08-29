#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import Image
import ImageDraw
import numpy as np
import struct
import time

def saveObjectCoord(save_name, pixel, coords):
    print save_name

    img_size = 500
    num_object = 24

    pixel = np.asarray(pixel, dtype = np.float32).reshape(img_size, img_size)
    pixel -= pixel.min()
    pixel *= 1.0 / pixel.max()
    pixel *= 255

 #   print coords
    coords = np.asarray(coords, dtype = np.float32).reshape(num_object, 4)
    coords *= img_size
    coords = coords.round().astype(np.int)
 #   print coords

    img = Image.fromarray(pixel.round().astype(np.uint8))

    draw = ImageDraw.Draw(img)
    for i in range(0, num_object):
        if (coords[i][0] >= 0) and (coords[i][1] >= 0) and coords[i][2] >=0 and coords[i][3] >= 0:
            draw.line((coords[i][0], coords[i][1], coords[i][2], coords[i][1]), fill=10)
            draw.line((coords[i][2], coords[i][1], coords[i][2], coords[i][3]), fill=10)
            draw.line((coords[i][2], coords[i][3], coords[i][0], coords[i][3]), fill=10)
            draw.line((coords[i][0], coords[i][3], coords[i][0], coords[i][1]), fill=10)

    img.save(save_name)

#因为训练集的数据是整数，而预测出来的是浮点数因此分开函数来求
def cutTrainObjectForClassify(save_name, pixel, coords):
    img_size = 500
    num_object = 24

    pixel = np.asarray(pixel, dtype = np.float32).reshape(img_size, img_size)
    pixel -= pixel.min()
    pixel *= 1.0 / pixel.max()
    pixel *= 255

    coords = np.asarray(coords, dtype = np.int).reshape(num_object, 4)
    output_imgs = np.zeros((num_object, img_size*img_size))

    img = Image.fromarray(pixel.round().astype(np.uint8))

    for i in range(0, num_object):
        if (coords[i][0] > 0) and (coords[i][1] > 0) and coords[i][2] > 0 and coords[i][3] > 0:
            img_crop = img.crop((coords[i][0], coords[i][1], \
                                 coords[i][2], coords[i][3]))
            img_new = Image.new('L', (img_size, img_size), 0)
            img_new.paste(img_crop, (coords[i][0], coords[i][1], \
                                     coords[i][2], coords[i][3]))

            img_data = np.asarray(img_new.getdata(), dtype = np.uint8).reshape(img_size*img_size)
            tmp_mean = np.empty(img_size*img_size)
            tmp_mean.fill(np.mean(img_data))
            img_mean = np.subtract(img_data, tmp_mean)
            output_imgs[i] = np.copy(img_mean)

    return output_imgs



