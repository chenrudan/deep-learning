#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
图像处理相关的一些函数
'''

import sys
import os
import Image
import numpy as np
import struct
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

classify_dict = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, \
                 'horse':5, 'sheep':6, 'aeroplane':7, 'bicycle':8, \
                 'boat':9, 'bus':10, 'car':11, 'motorbike':12, \
                 'train':13, 'bottle':14, 'char':15, 'dining table':16, \
                 'potted plant':17, 'sofa':18, 'tvmonitor':19}

def convertImgToArray(filename):
    img = Image.open(filename)
    img_channel = 0;
    if img.mode == "RGB":
        img_channel = 3
    elif img.mode == "L":
        img_channel = 1

    img_result = np.array(img).reshape(img_channel, img.size[1], img.size[0])
    img_ori = np.array(img)

    for i in range(0, img_channel):
        for j in range(0, img.size[0]):
            for k in range(0, img.size[1]):
                 img_result[i][k][j] = img_ori[k][j][i]
    return [img_result, img.size[0], img.size[1], img_channel]

'''
返回这个目录以及子目录下的所有图片。第一个返回值是图片的绝对路径
第二个是图片名称
'''
def getFilesOfDir(dir_name):
    #files只是图片的文件名，需要获取它的路径才能打开
    files_absolute_path = []
    for root, dirs, files in os.walk(dir_name):
        if dirs:
            for one_dir in dirs:
                for one_file in glob.glob(root + one_dir + '/*.'):
                    files.append(root + one_file)
        else:
            for one_file in files:
                files_absolute_path.append(root + one_file)

    return [files_absolute_path, files]


def meanOneImg(img_array, img_length, img_channel):
    mean_result = np.array(img_array).reshape(img_length*img_channel).astype(np.float32)
    for i in range(0, img_channel):
        avg = 0.0
        for j in range(0, img_length):
            avg = avg + mean_result[i*img_length+j]
        avg = avg / img_length
        for j in range(0, img_length):
            mean_result[i*img_length + j] = mean_result[i*img_length+j] - avg
    return mean_result


def parseVOCAnnotationsOneImg(xml_name):
    tree = ET.parse(xml_name)

    objects_info = []

    for l1_child in tree.iter(tag='object'):
        one_object = []
        for l2_child in l1_child:
            if l2_child.tag == 'name':
                one_object.append(classify_dict[l2_child.text])
            if l2_child.tag == 'bndbox':
                coordinate_dict = {}
                for l3_child in l2_child:
                       coordinate_dict[l3_child.tag] = l3_child.text

                one_object.append(int(coordinate_dict['xmin']))
                one_object.append(int(coordinate_dict['ymin']))
                one_object.append(int(coordinate_dict['xmax']))
                one_object.append(int(coordinate_dict['ymax']))

        objects_info.append(one_object)
    return objects_info

def supplyOneImg(img_array, dst_width, dst_height, ori_width, ori_height, \
                 channels):
    img_array = img_array.reshape(channels*ori_width*ori_height)
    dst_array = np.zeros(channels*dst_width*dst_height)
    print len(dst_array)
    print img_array
    for k in range(0, channels):
        for i in range(0, dst_width):
            for j in range(0, dst_height):
                if i < ori_width or j < ori_height:
                    dst_array[k*dst_width*dst_height + j*dst_width + i] \
                        = img_array[k*ori_width*ori_height + j*ori_width + i]
                else:
                    dst_array[k*dst_width*dst_height + j*dst_width+ i] = 0
    print dst_array
    return dst_array


'''
num_img + img_channel + img_width + img_height
+ num_object + object1_type + x_min + y_min + x_max + y_max + object2... + pixels
+ ...
'''
def convertVOCimgToBinary():
    img_dir = '/home/crd/work/deeplearning/data/VOCdevkit/VOC2012/JPEGImages/'
    annotator_dir = '/home/crd/work/deeplearning/data/VOCdevkit/VOC2012/Annotations/'
    [img_absolute_path, img_names] = getFilesOfDir(img_dir)

    img_absolute_width = 500
    img_absolute_height = 500
    img_channel = 3

    VOC_save = open('VOCdata.bin', 'w')
    packed_voc_data = struct.pack('i', len(img_absolute_path))
    packed_voc_data = packed_voc_data + struct.pack('iii', img_channel, \
                                    img_absolute_width, img_absolute_height)

   # for i in range(0, len(img_absolute_path)):
    for i in range(0, 1):
        [img_array, img_width, img_height, img_channel] \
                = convertImgToArray(img_absolute_path[i])

        print img_absolute_path[i]
        print img_height
        print img_width

        img_supply_array = supplyOneImg(img_array, img_absolute_width, \
                            img_absolute_height, img_width, img_height, \
                                        img_channel)
        tmp = Image.fromarray(img_supply_array)
        tmp.show()
        img_mean_array = meanOneImg(img_array, img_width*img_height, img_channel)
        #将图片补全成一样大小


        #解析图片对应的xml
        annotator_name = annotator_dir + os.path.splitext(img_names[i])[0] + '.xml'
        print annotator_name

        annotator_name = annotator_dir + '2007_000323' + '.xml'
        objects_info = parseVOCAnnotationsOneImg(annotator_name)

        packed_voc_data = packed_voc_data + struct.pack('i', len(objects_info))
        print packed_voc_data
        for j in range(0, len(objects_info)):
            packed_voc_data = packed_voc_data + struct.pack('iiiii', \
                                objects_info[j][0], objects_info[j][1], \
                                objects_info[j][2], objects_info[j][3], \
                                objects_info[j][4])
        print packed_voc_data




if __name__ == "__main__":
    convertVOCimgToBinary()
















