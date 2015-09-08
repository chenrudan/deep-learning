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
import matplotlib.pyplot as plt

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

        for one_file in files:
            print root + '/' + one_file
            one_file_prefix = basename + '_' + os.path.splitext(one_file)[0] + '_'

            if not os.path.exists(out_dir+os.path.splitext(one_file)[0]):
                os.makedirs(out_dir + os.path.splitext(one_file)[0])
            
            eng.c_elegans_segmentation(root+'/'+one_file, \
                    out_dir+os.path.splitext(one_file)[0] + '/' + one_file_prefix)

def drawHistOfImgSizeDistribute():
    main_dir = '/home/crd/work/c.elegans/data/dic_images_training/'
    num_types = 4
    all_larger_side = []
    class_name = []

    for root, dirs, files in os.walk(main_dir):
        larger_list = []  #保存每一张图片比较大的一条边
        basename = os.path.basename(root)
        
        for one_file in files:
            img = Image.open(root+'/'+one_file).convert('L')
            larger_list.append(max(img.size[0], img.size[1]))

        print basename

        larger_array = np.asarray(larger_list, dtype=np.int)

        if larger_array.size:
            all_larger_side.append(larger_array)
            class_name.append(basename)

    plt.figure()

    plt.hist([all_larger_side[0], all_larger_side[1], all_larger_side[2], all_larger_side[3]], 20, \
            label=[class_name[0], class_name[1], class_name[2], class_name[3]])
    plt.legend()
    plt.ylabel('image number')
    plt.xlabel('larger size')
    plt.title('size distribute of four class')

#    plt.savefig('/home/crd/work/c.elegans/data/imgsize_distribute.png')
    plt.show()

def getFilesAndLabelOfDir(dir_name):
    #files只是图片的文件名，需要获取它的路径才能打开
    files_absolute_path = []
    all_files = []
    label = []
    ori_img_idx = []  #从原图切割出来的原图id
    index = 0  
    ori_idx_value = 0   #保存了键值对的值
    ori_class_idx = 0  #保存了原来大图的类别
    fout = open('../record/segment_classification_mapping.txt', 'w')
    fout1 = open('../record/ori_img_idx_mapping.txt', 'w')
    fout2 = open('../record/ori_classification_mapping.txt', 'w')
    index_of_img_dict = {}
    class_of_ori_img_dict = {}

    for root, dirs, files in os.walk(dir_name):

        #在路径下只有文件夹，只有文件夹里面有图片
        if files:
            for one_file in files:
                files_absolute_path.append(root + '/'+ one_file)
                label.append(index)
                all_files.append(one_file)
        
                one_file_split = one_file.split('_')
                one_file_split.pop()
                idx_key = '_'.join(one_file_split)
                if not index_of_img_dict.has_key(idx_key):
                    index_of_img_dict[idx_key] = ori_idx_value
                    fout1.write(idx_key)
                    fout1.write("\t")
                    fout1.write(str(ori_idx_value))
                    fout1.write("\n")
                    ori_idx_value = ori_idx_value + 1

                one_file_split.pop()
                one_file_split.pop()
                class_key = '_'.join(one_file_split)
                if not class_of_ori_img_dict.has_key(class_key):
                    class_of_ori_img_dict[class_key] = ori_class_idx
                    fout2.write(class_key)
                    fout2.write("\t")
                    fout2.write(str(ori_class_idx))
                    fout2.write("\n")
                    ori_class_idx = ori_class_idx + 1


                ori_img_idx.append(index_of_img_dict[idx_key])


            fout.write(root)
            fout.write("\t")
            fout.write(str(index))
            fout.write("\n")
            index = index + 1


    return [files_absolute_path, label, all_files, ori_img_idx]

def supplyAndPackedOneImg(img_array, packed_data, class_label, ori_class_label, \
        dst_width, dst_height, ori_width, ori_height, ori_channel):
    
    img_supply = DIC_img_tools.supplyOneImg(img_array, dst_width, \
                    dst_height, ori_width, ori_height, ori_channel)
    
    img_mean = DIC_img_tools.meanOneImg(img_supply, dst_height, \
            dst_width, ori_channel)
    array_len = dst_width*dst_height*ori_channel
    img_mean_array = img_mean.reshape(array_len)
    
    packed_data = packed_data + struct.pack('i', ori_class_label)
    packed_data = packed_data + struct.pack('i', class_label)
    packed_data = packed_data + struct.pack('f'*array_len, *img_mean_array)

    return [packed_data, img_mean_array]

def chooseRange(area, size_list):
    for i in range(1, len(size_list)):
        if area < size_list[i]*size_list[i]*0.6:
            return i-1
        
    return len(size_list)

def supplyImgToDiffSize(img_main_dir, save_name, save_class_num):
    #图片补全成四种规格32，64，96，150
    [img_absolute_path, img_labels, img_names, img_idxs] \
            = getFilesAndLabelOfDir(img_main_dir)
    size_list = [32, 64, 96, 150, 400]

    #保存四个scale的二进制文件
    save_32 = open(save_name+'_32.bin', 'w')
    save_64 = open(save_name+'_64.bin', 'w')
    save_96 = open(save_name+'_96.bin', 'w')
    save_150 = open(save_name+'_150.bin', 'w')
    save_list = [save_32, save_64, save_96, save_150]

    num_list = [0, 0, 0 ,0]

    fout = open(save_class_num, 'w')
    save_class_dict = {}

    img_channel = 1
    for i in range(0, len(img_names)):
        img = Image.open(img_absolute_path[i]).convert('L')
        area = img.size[0]*img.size[1]
        k = chooseRange(area, size_list)
        if k < 4:
            num_list[k] = num_list[k] + 1

    for i in range(0, 4):
        print 'the number of size' + str(size_list[i]) +': '+ str(num_list[i])

    packed_32 = struct.pack('i', num_list[0])
    packed_64 = struct.pack('i', num_list[1])
    packed_96 = struct.pack('i', num_list[2])
    packed_150 = struct.pack('i', num_list[3])
   
    packed_list = [packed_32, packed_64, packed_96, packed_150]

    for i in range(0, 4):
        packed_list[i] = packed_list[i] + struct.pack('iii', img_channel, size_list[i], size_list[i])

    for i in range(0, len(img_names)):
        img = Image.open(img_absolute_path[i]).convert('L')
        img_array = DIC_img_tools.convertImgToArray(img, img_channel)
        dst_size = 0
        area = img.size[0]*img.size[1]

        k = chooseRange(area, size_list)

        if k < 4:
            dst_size = size_list[k]
            [packed_list[k], img_mean_array] = supplyAndPackedOneImg(img_array, \
                    packed_list[k], img_labels[i], img_idxs[i], \
                    dst_size, dst_size, img.size[0], \
                    img.size[1], img_channel)
            
#            pixel = np.asarray(img_mean_array, dtype = np.float32).reshape( \
#                    dst_size, dst_size)
#            pixel -= pixel.min()
#            pixel *= 1.0 / pixel.max()
#            pixel *= 255
#            tmp = Image.fromarray(pixel.astype(np.uint8))
#            tmp.save('./tmp/'+ str(size_list[k])+ '/'+ str(i) + img_names[i])
        
            if len(packed_list[k]) > 268000000:
                save_list.write(packed_list[k])
                packed_list[k] = ''

        else:
            print img_names[i]
            if not save_class_dict.has_key(img_idxs[i]):
                save_class_dict[img_idxs[i]] = 1
            else:
                save_class_dict[img_idxs[i]] = save_class_dict[img_idxs[i]] + 1
#            pixel = np.asarray(img_array, dtype = np.float32).reshape( \
#                    img.size[1], img.size[0])
#            tmp = Image.fromarray(pixel.astype(np.uint8))
#            tmp.save('./tmp/other/'+ str(i) + img_names[i])
    for i in range(0, 4):
        if packed_list[i]:
            save_list[i].write(packed_list[i])
            packed_list[i] = ''

    save_class_items = save_class_dict.items()
    for i in range(0, len(save_class_items)):
        fout.write(str(save_class_items[i][0]))
        fout.write("\t")
        fout.write(str(save_class_items[i][1]))
        fout.write("\n")
        
        


if __name__ == "__main__":
  #  drawHistOfImgSizeDistribute()
  # imgSegmentation()
    supplyImgToDiffSize('/home/crd/work/c.elegans/data/dic_images_training/', \
          '/home/crd/work/c.elegans/data/DIC_seg_train', \
          '../record/final_train_class_num.txt')
    supplyImgToDiffSize('/home/crd/work/c.elegans/data/dic_images_valid/', \
          '/home/crd/work/c.elegans/data/DIC_seg_valid', \
          '../record/final_valid_class_num.txt')

















