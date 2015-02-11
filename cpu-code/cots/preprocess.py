#!/usr/bin/python 

import numpy as np
import os
import Image  
import math

def preprocess(image, i, j):
	img_data = np.asarray(image[i][j].reshape(96,96))
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
	
def main():
	f = np.fromfile('../../data/stl10/train_X.bin', dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	preprocessed = []

	for i in range(0, 2400):
		for j in range(0, 3):
			pre_ele = preprocess(image, i, j)
			preprocessed.append(pre_ele.tolist())
	np.asarray(preprocessed).astype(np.float32).tofile('show.bin')
	#np.savetxt("show.txt" , np.asarray(preprocessed*255).astype(np.float32))


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

if __name__ == "__main__":
	main()
	
	
	
	
	
	
	
	
	
