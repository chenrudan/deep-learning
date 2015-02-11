#!/bin/bash

import numpy as np
import os
import Image  
import math
import re
import glob
import random

def preprocess(image):
	img_data = np.asarray(image).astype(np.uint8).reshape(96, 96)
	img_data = img_data - img_data.mean()
	img_data = img_data / img_data.std()
	width = img_data.shape[0]
	height = img_data.shape[1]
	fx, fy = np.meshgrid(np.linspace(-width/2, width/2-1, width), np.linspace(-height/2, height/2-1, height))
	rho = np.sqrt(fx * fx + fy * fy)
	f0 = 0.4 * np.mean([width, height])
	filt = rho * np.exp(- np.power(rho/f0, 4))
	If = np.fft.fft2(img_data)
	img_data = np.real(np.fft.ifft2(If * np.fft.fftshift(filt)))
	img_data = img_data / img_data.std()
	img_data = img_data - img_data.mean()
	img_data = img_data / np.sqrt(np.mean(np.power(img_data, 2)))
	img_data = np.sqrt(0.1) * img_data
	
	#img_data = scale_to_unit_interval(img_data)
	return img_data

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
	
def main():

	train_pixel = []
	train_label = []	
	test_pixel = []
	test_label = []
	
	train = []
	test = []
	index = 0
	list_dirs = os.walk('./')
	for root, dirs, files in list_dirs:		
		for img_dir in dirs:
			number = 0;
			for img_addr in glob.glob('./' + img_dir + '/*'):
				print img_addr
				im = Image.open(img_addr)
				im_resize = im.resize((96, 116), Image.ANTIALIAS)				
				region = (0, 10, 96, 106)
				im_crop = im_resize.crop(region)
				im_prep = preprocess(im_crop)
				
				if number < 8:
					train.append([im_prep, index])
				else:
					test.append([im_prep, index])
				number = number + 1
			index = index + 1
	random.shuffle(train)
	random.shuffle(test)
	for i in range(0, 320):
		train_pixel.append(train[i][0])
		train_label.append(train[i][1])
	for i in range(0, 80):
		test_pixel.append(test[i][0])
		test_label.append(test[i][1])
			
	np.asarray(train_pixel, dtype = np.float64).tofile('orlfaces_train.bin')
	np.asarray(train_label, dtype = np.uint8).tofile('orl_label_train.bin')
	np.asarray(test_pixel, dtype = np.float64).tofile('orlfaces_test.bin')
	np.asarray(test_label, dtype = np.uint8).tofile('orl_label_test.bin')
		
if __name__ == "__main__":
	main()
	
	
