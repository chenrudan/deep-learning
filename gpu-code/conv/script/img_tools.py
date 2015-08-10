#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
图像处理相关的一些函数
'''

import os
import Image
import numpy as np
import struct
from time import clock
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

classify_dict = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, \
                 'horse':5, 'sheep':6, 'aeroplane':7, 'bicycle':8, \
                 'boat':9, 'bus':10, 'car':11, 'motorbike':12, \
                 'train':13, 'bottle':14, 'chair':15, 'diningtable':16, \
                 'pottedplant':17, 'sofa':18, 'tvmonitor':19}


def convertImgToArray(filename):
    img = Image.open(filename)
    img_channel = 0;
    if img.mode == "RGB":
        img_channel = 3
    elif img.mode == "L":
        img_channel = 1

    img_result = np.array(img).reshape(img.size[1]*img.size[0], img_channel).transpose()
    img_result = img_result.reshape(img_channel, img.size[1], img.size[0])

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


def meanOneImg(img_array, img_width, img_height, img_channel):

    img_length = img_width*img_height
    mean_result_array = np.array(img_array).reshape(img_channel, img_length).astype(np.float32)
    for i in range(0, img_channel):
        mean_value = np.mean(mean_result_array[i])
        mean_result_array[i] = mean_result_array[i] - np.ones(img_length)*mean_value

    return mean_result_array.reshape(img_channel, img_height, img_width)


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
                try:
                    one_object.append(int(coordinate_dict['xmin']))
                except ValueError:
                    one_object.append(int(float(coordinate_dict['xmin'])))
                try:
                    one_object.append(int(coordinate_dict['ymin']))
                except ValueError:
                    one_object.append(int(float(coordinate_dict['ymin'])))
                try:
                    one_object.append(int(coordinate_dict['xmax']))
                except ValueError:
                    one_object.append(int(float(coordinate_dict['xmax'])))
                try:
                    one_object.append(int(coordinate_dict['ymax']))
                except ValueError:
                    one_object.append(int(float(coordinate_dict['ymax'])))

        objects_info.append(one_object)
    return objects_info


def supplyOneImg(img_array, dst_width, dst_height, ori_width, ori_height, \
                 channels):
    dst_array = np.zeros((channels, dst_height, dst_width))
    dst_array[0:channels, 0:ori_height, 0:ori_width] = img_array
    return dst_array

def packVOCOneImg(packed_voc, objects_info, num_object, img_supply_array, \
			array_len, VOC_save):
	#打包object个数
	packed_voc = packed_voc + struct.pack('i', len(objects_info))
	num_object = num_object + len(objects_info)
    #打包每个object对应的左上和右下坐标
	for j in range(0, len(objects_info)):
		packed_voc = packed_voc + struct.pack('iiiii', \
		                        objects_info[j][0], objects_info[j][1], \
                                objects_info[j][2], objects_info[j][3], \
                                objects_info[j][4])
    #打包补全图片像素值
	packed_voc = packed_voc + struct.pack('f'*array_len, \
                           *img_supply_array)
		
	if len(packed_voc) > 268000000:
		VOC_save.write(packed_voc)
		packed_voc = ''
	
	return [packed_voc, num_object]
	

'''
num_img + img_channel + img_width + img_height
+ num_object + object1_type + x_min + y_min + x_max + y_max + object2... + pixels
+ ...
'''
def convertVOCimgToBinary():
	img_dir = '/home/crd/data/VOCdevkit/VOC2012/JPEGImages/'
	annotator_dir = '/home/crd/data/VOCdevkit/VOC2012/Annotations/'
	[img_absolute_path, img_names] = getFilesOfDir(img_dir)

	img_absolute_width = 500
	img_absolute_height = 500
	img_channel = 3

	VOC_train_save = open('/home/crd/data/VOCdevkit/VOC2012/VOC_train_data.bin', 'w')
	VOC_valid_save = open('/home/crd/data/VOCdevkit/VOC2012/VOC_valid_data.bin', 'w')

	num_train = len(img_absolute_path) * 9 / 10;
	num_valid = len(img_absolute_path) - num_train;

	packed_voc_train = struct.pack('i', num_train)
	packed_voc_train = packed_voc_train + struct.pack('iii', img_channel, \
                                    img_absolute_width, img_absolute_height)

	packed_voc_valid = struct.pack('i', num_valid)
	packed_voc_valid = packed_voc_valid + struct.pack('iii', img_channel, \
                                    img_absolute_width, img_absolute_height)

	print 'image number is: ' + str(len(img_absolute_path))
	print 'train number is: ' + str(num_train)
	print 'valid number is: ' + str(num_valid)
	print 'unified image width is: ' + str(img_absolute_width)
	print 'unified image height is: ' + str(img_absolute_height)
	print 'convert voc image to binary file.'
	print 'start processing...'

	num_train_object = 0
	num_valid_object = 0

	packed_array_len = img_channel*img_absolute_width*img_absolute_height

	for i in range(0, len(img_absolute_path)):
#	for i in range(0, 10):
		start=clock()
		[img_array, img_width, img_height, img_channel] \
                = convertImgToArray(img_absolute_path[i])

		print 'this image id is: ' + str(i)	
		print 'this image width is: ' + str(img_width)
		print 'this image height is: ' + str(img_height)

		img_mean_array = meanOneImg(img_array, img_width, img_height, img_channel)
        #将图片补全成一样大小
		img_supply_array = supplyOneImg(img_mean_array, img_absolute_width, \
                            img_absolute_height, img_width, img_height, \
                                        img_channel)
		if i < 10:
			tmp = Image.fromarray(img_supply_array[0].astype(np.uint8))
			tmp.save('./tmp/'+img_names[i])

        #解析图片对应的xml
		annotator_name = annotator_dir + os.path.splitext(img_names[i])[0] + '.xml'
		print annotator_name

		objects_info = parseVOCAnnotationsOneImg(annotator_name)
		
		img_supply_array = img_supply_array.reshape(img_channel*img_absolute_height*img_absolute_width)

		if i < num_train:
			[packed_voc_train, num_train_object] = packVOCOneImg(packed_voc_train, \
					objects_info, num_train_object, img_supply_array, \
					packed_array_len, VOC_train_save)	
		else:
			[packed_voc_valid, num_valid_object] = packVOCOneImg(packed_voc_valid, \
					objects_info, num_valid_object, img_supply_array, \
					packed_array_len, VOC_valid_save)	

		end = clock()
		print str(end-start)+' seconds'
	
	print 'train objects num is: ' + str(num_train_object)
	print 'valid objects num is: ' + str(num_valid_object)

	if packed_voc_train:
	    VOC_train_save.write(packed_voc_train)
	if packed_voc_valid:
	    VOC_valid_save.write(packed_voc_valid)

	VOC_train_save.close()
	VOC_valid_save.close()



if __name__ == "__main__":
    convertVOCimgToBinary()
















