#!/usr/bin/python

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
#import cv2


def save_w_to_img(filename, channel, filter_size, out_addr):
	w_ori = np.fromfile(filename, np.float32);
	w = w_ori.reshape(channel, filter_size, filter_size);

	plt.figure(1)
	for i in range(0, channel):
		unit_w = scale_to_unit_interval(w[i])
		tmp = Image.fromarray(unit_w).convert('L')
	 	plt.subplot(channel / 10, channel % 10, i % 10)
		plt.imshow(tmp, 'gray')
		
	plt.show()
	#plt.savefig(out_addr + 'cnn1.png')
	 # tmp.save( out_addr + str(i) +'.png')

def scale_to_unit_interval(ndar, eps=1e-8):
	""" Scales all values in the ndarray ndar to be between 0 and 1 """
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar*255

def main():
	cnn1_channel = 20;
	cnn1_filter_size = 5;
	cnn2_channel = 50;
	cnn2_filter_size = 4;

	save_w_to_img('../pars/cnn1_w_t1.bin', cnn1_channel, cnn1_filter_size, './pic/cnn1/')


if __name__ == "__main__":
	main()
