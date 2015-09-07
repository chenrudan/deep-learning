#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
图像处理相关的一些函数
'''

import os
import glob
import Image
import numpy as np
import struct
import DIC_img_tools
import cv2
import matlab.engine

def imgSegmentation():

    main_dir = '/home/crd/work/c.elegans/data/dic_images_full/'
    out_main_dir = '/home/crd/work/c.elegans/data/dic_images_segmentation/'

    eng = matlab.engine.start_matlab()
    for root, dirs, files in os.walk(main_dir):
        #返回当前目录名
        basename = os.path.basename(root)
        #对应的输出的目录名
        out_dir = out_main_dir + basename + '/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in files:
            print root + '/' + file
            file_prefix = basename + '_' + os.path.splitext(file)[0] + '_'

            if not os.path.exists(out_dir+os.path.splitext(file)[0]):
                os.makedirs(out_dir + os.path.splitext(file)[0])
            
            eng.c_elegans_segmentation(root+'/'+file, \
                    out_dir+os.path.splitext(file)[0] + '/' + file_prefix)



if __name__ == "__main__":
    imgSegmentation()

















