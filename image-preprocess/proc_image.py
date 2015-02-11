#!/usr/bin/python
#Filename: proc_image.py

import numpy as np
import os
import Image  
import math
import types
import sys

class BiggerThanZeroError(Exception):
	'''Value must be bigger than zero.'''
	def __init__(self, var):
		Exception.__init__(self)
		self.var = var

class ProcImage(object):
	'''Represents part of functions of image processing, including convert
	RGB image to grey image, scale and crop iamge size, image whiten ... 
	
	Attributes:
	
	'''
	
	def __init__(self):
		'''Initializes class ProcImage.'''
	
	def whiten(self, image):
		'''Whites one input image with fft transform.
		
		Args:
			image: Can be a list or Image object which represents the 
				pixels value of one image. 
			width: Width of image.
			height: Height of image.
		
		Returns:
			A numpy array contains whiten pixels value.
			
		Raises:
			TypeError: An error occurred while width or height is not integer.
			ValueError: An error occurred while width or height is less than 0.
		'''				
		(width, height) = image.size
		img_data = np.asarray(image).astype(np.uint8).reshape(width, height)
		img_data = img_data - img_data.mean()
		img_data = img_data / img_data.std()
		fx, fy = np.meshgrid(np.linspace(-width/2, width/2-1, width), 
							 np.linspace(-height/2, height/2-1, height))
		rho = np.sqrt(fx * fx + fy * fy)
		f0 = 0.4 * np.mean([width, height])
		filt = rho * np.exp(- np.power(rho/f0, 4))
		If = np.fft.fft2(img_data)
		img_data = np.real(np.fft.ifft2(If * np.fft.fftshift(filt)))
		img_data = img_data / img_data.std()
		img_data = img_data - img_data.mean()
		img_data = img_data / np.sqrt(np.mean(np.power(img_data, 2)))
		img_data = np.sqrt(0.1) * img_data
		return img_data
		
	def zoom(self, im, scale = 1, crop_ul = 0, crop_ur = 0, crop_ll = 0, crop_lr = 0):
		'''Zoom and crop image
		
		Args:
			im: An Image object which represents the 
				pixels value of one image.
			scale: Zoom image.
		 	crop_ul: Crop image with start at upper left size.
		 	crop_ur: Crop image with start at upper right size.
		 	crop_ll: Crop image with start at lower left size.
		 	crop_lr: Crop image with start at lower right size.
		 	
		Returns:
			Returns an Image object which contain image pixels. 
		'''
		(width, height) = im.size
		width = width / scale
		height = height / scale		
		im_resize = im.resize((width, height), Image.ANTIALIAS)
		region = (crop_ul, crop_ur, width - crop_ll, width - crop_lr)
		im_crop = im_resize.crop(region)
		return im_crop
		
	def convertImgTobin(self, addr, save_name, addr_type = 'IMAGE', dtype = np.uint8, 
						file_num = None, scale = 1, crop_ul = 0, crop_ur = 0, 
						crop_ll = 0, crop_lr = 0, white = False):
		'''Converts image to binary file.
		
		 Args:
		 	addr: Address of input, can be an image of dir of floder.
		 	save_name: Process image and save into binary.
		 	addr_type: Show the type of addr which can be dir or image.
		 		'DIR': Dir address.
		 		'IMAGE': Image address.
		 	dtype: The save type of object of this address. 
		 	file_num: Handle multiple files including folders which is only valid 
		 		while addr_type is 'DIR'.
		 	scale: Zoom image.
		 	crop_ul: Crop image with start at upper left size.
		 	crop_ur: Crop image with start at upper right size.
		 	crop_ll: Crop image with start at lower left size.
		 	crop_lr: Crop image with start at lower right size.
		 	white: Need whiten or not.
		 	
		 Returns: 
		 	Return boolean means whether the convertion is success or not.
		 	
		 Raises:		 
		 '''
		try:
			if addr_type != 'IMAGE' and addr_type != 'DIR':
		 		raise ValueError
		 	elif scale <= 0:
		 		raise BiggerThanZeroError('scale')
		 	elif crop_ul < 0 or crop_ur < 0 or crop_ll < 0 or crop_lr < 0:
		 		raise BiggerThanZeroError('crop')
		except ValueError:
			print "address type must be 'IMAGE' or 'DIR'"
			sys.exit()
		except BiggerThanZeroError as x:
		 	print x.var + " must not less than 0."
			sys.exit()
			
		pixel = []
			 
		if addr_type == 'IMAGE':
			im = Image.open(addr)
			im_crop = self.zoom(im, scale, crop_ul, crop_ur, crop_ll, crop_lr)
			if not white:
				im_white = self.whiten(im_crop)
			pixel.append(im_white)
		elif addr_type == 'DIR':
			list_dirs = os.walk(addr)
			for root, dirs, images_addr in list_dirs:
				for im_addr in images_addr:
					try:
						im = Image.open(im_addr)
					except IOError:
						pass
					im_crop = self.zoom(im, scale, crop_ul, crop_ur, crop_ll, crop_lr)
					if not white:
						im_white = self.whiten(im_crop)
					pixel.append(im_white)
		print save_name
		np.asarray(pixel, dtype = np.float32).tofile(save_name)
					
					

		
		 
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
			
		
