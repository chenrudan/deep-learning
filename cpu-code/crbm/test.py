#!/usr/bin/python
#Filename:test.py
import string
import numpy as np
from hello import tile_raster_images
import PIL.Image
import theano

def readlayer():
	layerlist=[]
	f=open('./layer1.txt','r')
	s=f.read()
	b=s.split('\n')
	del b[-1]
	for element in b:
		layerlist.append(string.atof(element))
	a=np.array(layerlist,dtype=np.float64).reshape(784,500)
	return a

def plotimage(a):
	image = PIL.Image.fromarray(tile_raster_images(a.T,img_shape=(28,28),tile_shape=(10,10),tile_spacing=(1,1)))
	image.save('layer1.png')

def plotlayer():
    a = readlayer()
    plotimage(a)

def main():
	a=readlayer()
	plotimage(a)

if __name__=='__main__':
	main()
