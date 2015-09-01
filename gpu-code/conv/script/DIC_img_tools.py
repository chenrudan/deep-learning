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
from time import clock
import random

angle = {0:0, 1:30, 2:60, 3:120, 4:150}

def convertImgToArray(img, img_channel):

    img_result = np.array(img).reshape(img.size[1]*img.size[0], img_channel).transpose()
    img_result = img_result.reshape(img_channel, img.size[1], img.size[0])

    return img_result


'''
返回这个目录以及子目录下的所有图片。第一个返回值是图片的绝对路径
第二个是图片名称
'''
def getFilesAndLabelOfDir(dir_name):
    #files只是图片的文件名，需要获取它的路径才能打开
    files_absolute_path = []
    all_files = []
    label = []
    index = 0
    fout = open('classification_mapping.txt', 'w')
    for root, dirs, files in os.walk(dir_name):

        #在路径下只有文件夹，只有文件夹里面有图片
        if files:
            for one_file in files:
                files_absolute_path.append(root + '/'+ one_file)
                label.append(index)
                all_files.append(str(index) + one_file)
            fout.write(root)
            fout.write("\t")
            fout.write(str(index))
            fout.write("\n")
            index = index + 1



    return [files_absolute_path, label, all_files]


def meanOneImg(img_array, img_width, img_height, img_channel):

    img_length = img_width*img_height
    mean_result_array = np.array(img_array).reshape(img_channel, img_length).astype(np.float32)
    for i in range(0, img_channel):
        mean_value = np.mean(mean_result_array[i])
        mean_result_array[i] = mean_result_array[i] - np.ones(img_length)*mean_value

    return mean_result_array.reshape(img_channel, img_height, img_width)


def supplyOneImg(img_array, dst_width, dst_height, ori_width, ori_height, \
                 channels):
    dst_array = np.zeros((channels, dst_height, dst_width))
    dst_array[0:channels, 0:ori_height, 0:ori_width] = img_array
    return dst_array

def transformImg(img, idx):
    if idx % 4 == 1 or idx % 4 == 3:
       img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if idx % 4 == 2 or idx % 4 == 3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    img = img.rotate(angle[idx%5])
    return img

def packedLabelAndPixel(packed_data, label, array_len, img_array):
    packed_data = packed_data + struct.pack('i', label)
    packed_data = packed_data + struct.pack('f'*array_len, \
                                    *img_array)
    return packed_data

def convertDICimgToBinary():

    img_dir = '../../data/2015-08-22'

    [img_absolute_path, img_labels, img_names] = getFilesAndLabelOfDir(img_dir)

    print 'image number is: ' + str(len(img_absolute_path))
    print 'convert DIC image to binary file.'
    print 'start processing...'

    img_absolute_width = 1280
    img_absolute_height = 960
    img_channel = 1

    num_trans = 20
    num_class = 7

    num_valid_each_class = 2;
    DIC_train_save = open('DIC_train_data.bin', 'w')
    DIC_valid_save = open('DIC_valid_data.bin', 'w')

    packed_train_data = struct.pack('i', len(img_absolute_path)*num_trans \
                                    -num_valid_each_class*num_class*num_trans)
    packed_valid_data = struct.pack('i', num_valid_each_class*num_class*num_trans)

    packed_train_data = packed_train_data + struct.pack('iii', img_channel, \
                                    img_absolute_width, img_absolute_height)
    packed_valid_data = packed_valid_data + struct.pack('iii', img_channel, \
                                    img_absolute_width, img_absolute_height)

    array_len = img_channel*img_absolute_width*img_absolute_height

    last_label = 0
    save_valid_time = 0   #累计保存了2个valid就开始保存train

    train_list = []
    valid_list = []

    for i in range(0, len(img_absolute_path)):
 #   for i in range(0, 30):
        print img_absolute_path[i]
        img = Image.open(img_absolute_path[i]).convert('L')

        print 'this image id is: ' + str(i)
        print 'this image label is: ' + str(img_labels[i])
        for j in range(0, num_trans):
            img_trans = transformImg(img, j)

            img_array = convertImgToArray(img_trans, img_channel)

            img_mean_array = meanOneImg(img_array, img.size[1], \
                                        img.size[0], img_channel)

        #    print img_mean_array

            pixel = np.asarray(img_mean_array[0], dtype = np.float32).reshape( \
                                        img_absolute_height, img_absolute_width)
            if i == 1:
                pixel -= pixel.min()
                pixel *= 1.0 / pixel.max()
                pixel *= 255
                tmp = Image.fromarray(pixel.astype(np.uint8))
                tmp.save('./tmp/'+ str(j) + img_names[i])

            img_mean_array = img_mean_array.reshape( \
                            img_channel*img_absolute_height*img_absolute_width)

            tmp = []
            tmp.append(img_labels[i])
            tmp.append(img_mean_array)

            if img_labels[i] != last_label:
                last_label = img_labels[i]
                save_valid_time = 0

            if img_labels[i] == last_label and save_valid_time < 2:
                valid_list.append(tmp)
                print "valid image idx: " + str(i) + "      transform idx: " + str(j)
                if j == 19:
                    save_valid_time = save_valid_time + 1
            else:
                print "train image idx: " + str(i) + "      transform idx: " + str(j)
                train_list.append(tmp)


    print len(train_list)
    print len(valid_list)
  #  print valid_list

    random.shuffle(train_list)
    random.shuffle(valid_list)
#    print train_list


    for i in range(0, num_valid_each_class*num_trans*num_class):
        packed_valid_data = packedLabelAndPixel(packed_valid_data, \
                        valid_list[i][0], array_len, valid_list[i][1])
        if len(packed_valid_data) > 268000000:
            DIC_valid_save.write(packed_valid_data)
            packed_valid_data = ''

    for i in range(0, num_trans*len(img_absolute_path) \
                   - num_valid_each_class*num_trans*num_class):
        packed_train_data = packedLabelAndPixel(packed_train_data, \
                        train_list[i][0], array_len, train_list[i][1])
        if len(packed_train_data) > 268000000:
            DIC_train_save.write(packed_train_data)
            packed_train_data = ''

    if packed_train_data:
        DIC_train_save.write(packed_train_data)
    if packed_valid_data:
        DIC_valid_save.write(packed_valid_data)

    DIC_train_save.close()
    DIC_valid_save.close()

if __name__ == "__main__":
    convertDICimgToBinary()
















