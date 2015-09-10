#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding = utf-8

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
import random

class_of_ori_img_dict = {'1.control':0, '2.lva(or_let)':1, '3.ste(&.pvl)':2, \
        '4.emb':3, '5.leo':4, '6.leo(or_leo++)':5, '7.Adl(&.bag)':6}
class_of_seg_img_dict = {'adult':0, 'larva':1, 'embryo':2, 'background':3}

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

def getFilesAndLabelOfDir(dir_name, ori_img_idx_record):
    #files只是图片的文件名，需要获取它的路径才能打开
    files_absolute_path = []
    all_files = []
    label = []
    ori_img_idx = []  #从原图切割出来的原图id
    ori_img_label = []
    ori_idx_value = 0   #保存了键值对的值
    fout1 = open(ori_img_idx_record, 'w')
    index_of_img_dict = {}

    for root, dirs, files in os.walk(dir_name):

        #在路径下只有文件夹，只有文件夹里面有图片
        if files:
            for one_file in files:
                files_absolute_path.append(root + '/'+ one_file)
                basename = os.path.basename(root)
                label.append(class_of_seg_img_dict[basename])
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

                print one_file_split
                one_file_split.pop()
                one_file_split.pop()
                class_key = '_'.join(one_file_split)

                ori_img_idx.append(index_of_img_dict[idx_key])
                ori_img_label.append(class_of_ori_img_dict[class_key])
    fout1.close()

    return [files_absolute_path, label, all_files, ori_img_idx, ori_img_label]

def chooseRange(area, size_list):
    for i in range(1, len(size_list)):
        if area < size_list[i]*size_list[i]*0.6:
            return i-1
        
    return len(size_list)-1

def createBackgroudImg():
    i = 0
    for root, dirs, files in os.walk('/home/crd/work/c.elegans/data/dic_images_full/2.lva(or let)'):
        for one_file in files:
            img = Image.open(root+'/'+one_file).convert('L')
            img_array = DIC_img_tools.convertImgToArray(img, 1)
            img_supply = DIC_img_tools.supplyOneImg(img_array, 320, \
                    320, img.size[0], img.size[1], 1)
            pixel = np.asarray(img_supply).reshape(320, 320)
            tmp = Image.fromarray(pixel.astype(np.uint8))
            tmp.save('/home/crd/work/c.elegans/data/dic_images_training/background/back' + str(i) + '.png')
            i = i + 1

def drawHistOfDiffClassDistribute(img_main_dir, ori_img_idx_record):
    [img_absolute_path, img_labels, img_names, img_idxs, ori_img_label] \
            = getFilesAndLabelOfDir(img_main_dir, ori_img_idx_record)

    diff_ori_class = [[], [], [], [], [], [], []] #6种性状中每一类的每一张图中有多少四种类型虫子
    ori_img_map_worm_type_dict = {} #一张原图id对应的四个类型虫子个数
    ori_img_map_phenotype_dict = {}
   
    print len(img_names)
    for i in range(0, len(img_names)):
        if not ori_img_map_worm_type_dict.has_key(img_idxs[i]):
            ori_img_map_worm_type_dict[img_idxs[i]] = [0,0,0,0]
            ori_img_map_phenotype_dict[img_idxs[i]] = ori_img_label[i]
        ori_img_map_worm_type_dict[img_idxs[i]][img_labels[i]] = \
                ori_img_map_worm_type_dict[img_idxs[i]][img_labels[i]] + 1
        
       # print "img class " + str(img_labels[i])
    print len(ori_img_map_worm_type_dict)

    for i in range(0, len(ori_img_map_worm_type_dict)):
        diff_ori_class[ori_img_map_phenotype_dict[i]].append(ori_img_map_worm_type_dict[i])
    diff_worm_class = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]
    print diff_ori_class
    for i in range(0, len(diff_ori_class)):
        tmp = np.asarray(diff_ori_class[i]).reshape(len(diff_ori_class[i]), 4)
        tmp = tmp.transpose()
        for j in range(0, 3):
            diff_worm_class[i][j] = tmp[j]

    print diff_worm_class
          
    ori_class_name = ['1.control', '2.lva(or_let)', '3.ste(&.pvl)', \
        '4.emb', '5.leo', '6.leo(or_leo++)', '7.Adl(&.bag)']

    plt.figure()
    for i in range(0, len(diff_worm_class)):
        plt.subplot(2, 4, i+1)
        plt.hist(diff_worm_class[i], label=['adult', 'larva', 'embryo'])
        plt.legend()

        plt.ylabel('number of images')
        plt.xlabel('number of four class')
        plt.title(ori_class_name[i])

    plt.savefig('../record/manual_classify.png')
    plt.show()

def supplyImgToDiffSize(img_main_dir, save_name, ori_img_idx_record):
    #图片补全成四种规格32，64，96，160, 320
    [img_absolute_path, img_labels, img_names, img_idxs, ori_img_label] \
            = getFilesAndLabelOfDir(img_main_dir, ori_img_idx_record)
    size_list = [32, 64, 96, 160, 320]

    #保存四个scale的二进制文件
    save_32 = open(save_name+'_32.bin', 'w')
    save_64 = open(save_name+'_64.bin', 'w')
    save_96 = open(save_name+'_96.bin', 'w')
    save_160 = open(save_name+'_160.bin', 'w')
    save_320 = open(save_name+'_320.bin', 'w')
    save_list = [save_32, save_64, save_96, save_160, save_320]

    num_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    num_expand = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    img_channel = 1
    for i in range(0, len(img_names)):
        img = Image.open(img_absolute_path[i]).convert('L')
        area = img.size[0]*img.size[1]
        k = chooseRange(area, size_list)
        num_list[k][img_labels[i]%4] = num_list[k][img_labels[i]%4] + 1

    fout = open(ori_img_idx_record, 'a')
    for i in range(0, len(num_list)):
        max_num = max(num_list[i])
        min_num = max_num
        min_idx = 0
#将个数中的0排除掉，然后求比最大值不少于20倍的最小值，以这个最小值为基准来扩大数据集
        for j in range(0, len(num_list[i])):
            if num_list[i][j] < min_num and num_list[i][j] > (max_num/20) or min_num is 0:
                min_num = num_list[i][j]
                min_idx = j

        for j in range(0, len(num_list[i])):
            if j == min_idx:
                num_expand[i][j] = 20*num_list[i][j]
            elif num_list[i][j] is not 0 and num_list[i][j] > (max_num/20):
                num_expand[i][j] = int(round(20.0*num_list[i][min_idx] / num_list[i][j]))*num_list[i][j]
            elif num_list[i][j] < (max_num/20):
                num_expand[i][j] = num_list[i][j]

        fout.write('size ' + str(size_list[i]) + ' original img number\texpand img number\n')    
        for j in range(0, len(num_list[i])):
            print 'the number of size ' + str(num_list[i][j]) +' ... '+ str(num_expand[i][j])
            fout.write(str(num_list[i][j]))
            fout.write("\t")
            fout.write(str(num_expand[i][j]))
            fout.write("\n")

        print '--------'

    fout.close()
    packed_32 = struct.pack('i', sum(num_expand[0]))
    packed_64 = struct.pack('i', sum(num_expand[1]))
    packed_96 = struct.pack('i', sum(num_expand[2]))
    packed_160 = struct.pack('i', sum(num_expand[3]))
    packed_320 = struct.pack('i', sum(num_expand[4]))

    print sum(num_expand[0])
    print sum(num_expand[1])
    print sum(num_expand[2])
    print sum(num_expand[3])
    print sum(num_expand[4])
   
    packed_list = [packed_32, packed_64, packed_96, packed_160, packed_320]

    train_list_32 = []
    train_list_64 = []
    train_list_96 = []
    train_list_160 = []
    train_list_320 = []

    train_list = [train_list_32, train_list_64, train_list_96, train_list_160, train_list_320]

    for i in range(0, len(size_list)):
        packed_list[i] = packed_list[i] + struct.pack('iii', img_channel, size_list[i], size_list[i])

    for i in range(0, len(img_names)):
        img = Image.open(img_absolute_path[i]).convert('L')
        area = img.size[0]*img.size[1]
        k = chooseRange(area, size_list)
        dst_size = size_list[k]
        array_len = dst_size*dst_size*img_channel
        for j in range(0, num_expand[k][img_labels[i]%4] / max(num_list[k][img_labels[i]%4], 1)):

            img_array = DIC_img_tools.convertImgToArray(img, img_channel)

            img_supply = DIC_img_tools.supplyOneImg(img_array, dst_size, \
                    dst_size, img.size[0], img.size[1], img_channel, 140)
    

            pixel = np.asarray(img_supply).reshape(dst_size, dst_size)
            tmp = Image.fromarray(pixel.astype(np.uint8))
            img_trans = DIC_img_tools.transformImg(tmp, j)

            img_array = DIC_img_tools.convertImgToArray(img_trans, img_channel)

            img_array = img_array.reshape(array_len)
            img_array = img_array - np.mean(img_array)
            img_array = img_array / np.std(img_array)

            tmp = []
            tmp.append(img_idxs[i])
            tmp.append(ori_img_label[i])
            tmp.append(img_labels[i])
            tmp.append(img_array)

            train_list[k].append(tmp)

    for i in range(0, len(size_list)):
        random.shuffle(train_list[i])

    for i in range(0, len(size_list)):
        print len(train_list[i])
        for j in range(0, len(train_list[i])):
            array_len = size_list[i]*size_list[i]*img_channel

            packed_list[i] = packed_list[i] + struct.pack('i', train_list[i][j][0])
            packed_list[i] = packed_list[i] + struct.pack('i', train_list[i][j][1])
            packed_list[i] = packed_list[i] + struct.pack('i', train_list[i][j][2])
            packed_list[i] = packed_list[i] + struct.pack('f'*array_len, *train_list[i][j][3])
           
            pixel = np.asarray(train_list[i][j][3], dtype = np.float32).reshape( \
                    size_list[i], size_list[i])
            pixel -= pixel.min()
            pixel *= 1.0 / pixel.max()
            pixel *= 255
            tmp = Image.fromarray(pixel.astype(np.uint8))
            tmp.save('./tmp/' + str(size_list[i]) + '/'+ str(i) + '_' + str(j) + '.png')
        
            if len(packed_list[i]) > 268000000:
                save_list.write(packed_list[i])
                packed_list[i] = ''
        save_list[i].write(packed_list[i])
        packed_list[i] = ''

    for i in range(0, len(packed_list)):
        if packed_list[i]:
            save_list[i].write(packed_list[i])

    save_32.close()
    save_64.close()
    save_96.close()
    save_160.close()
    save_320.close()

        


if __name__ == "__main__":
  #  drawHistOfImgSizeDistribute()
  # imgSegmentation()
  #  createBackgroudImg()

    supplyImgToDiffSize('/home/crd/work/c.elegans/data/dic_images_training/', \
          '/home/crd/work/c.elegans/data/DIC_seg_train', \
         '../record/ori_img_idx_mapping_train.txt')
    supplyImgToDiffSize('/home/crd/work/c.elegans/data/dic_images_valid/', \
          '/home/crd/work/c.elegans/data/DIC_seg_valid', \
          '../record/ori_img_idx_mapping_valid.txt')
#    drawHistOfDiffClassDistribute('/home/crd/work/c.elegans/data/dic_images_training/', \
#         '../record/ori_img_idx_mapping_train.txt')
    

















