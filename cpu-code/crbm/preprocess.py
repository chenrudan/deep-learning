
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
	
	img_data = scale_to_unit_interval(img_data)
	img = Image.fromarray((img_data*255.0).astype(np.uint8))
	#img.save(str(i+j) +'.png')
	return img_data
	

def main():
	f = np.fromfile('../../../../crd/deeplearning/data/stl10/train_X.bin', dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	preprocessed = []

	for i in range(0, 100):
		for j in range(0, 3):
			pre_ele = preprocess(image, i, j)
			preprocessed.append(pre_ele.tolist())
	np.asarray(preprocessed).astype(np.float32).tofile('show.bin')			
	f2 = np.fromfile('show.bin', dtype = np.float32)
	image2 = np.array(f2).reshape((100, 3, 96*96))
	print image2
	
"""	for k in range(0, 1):
		m =[]
		for i in range(0,96):
			n = []
			for j in range(0,96):
				n.append([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]])
			m.append(n)
		f = np.asarray(m, dtype = np.uint8)
		img = Image.fromarray(f,'RGB')
		img.save('unpreprocess' + str(k) +'.png')
	
	for k in range(0, 1):
		b =[]
		for i in range(0,96):
			a = []
			for j in range(0,96):
				a.append(np.dot(([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]]), [0.299, 0.587, 0.144]))
			b.append(a)
		c = np.asarray(b, dtype = np.uint8)
		img = Image.fromarray(c)
		img.save('unpreprocess' + str(k) +'.png')
		pre_ele = preprocess(c)
		pre_ele = scale_to_unit_interval(pre_ele)
		img2 = Image.fromarray((pre_ele*255).astype(np.uint8))
		img2.save('preprocess' + str(k) +'.png')"""	
		
"""	np.asarray(preprocessed*255).astype(np.uint8).tofile('preprocessed.bin')
	print np.shape(preprocessed)
	f2 = np.fromfile('preprocessed.bin', dtype = np.uint8)
	image2 = np.array(f2, dtype = np.uint8).reshape((20, 3, 96*96)) """

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
	
	
def test():
	f = np.fromfile('../../../../crd/deeplearning/data/stl10/train_X.bin', dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	preprocessed = []
	for k in range(0, 1):
		b =[]
		for i in range(0,96):
			a = []
			for j in range(0,96):
				a.append(np.dot(([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]]), [0.299, 0.587, 0.144]))
			b.append(a)
		c = np.asarray(b, dtype = np.uint8)
		img = Image.fromarray(c)
		img.save("test.png")
		pre_ele = preprocess(c)
		pre_ele = scale_to_unit_interval(pre_ele)
		img2 = Image.fromarray((pre_ele*255).astype(np.uint8))
		img2.save('preprocess.png')

if __name__ == "__main__":
	main()
	#test()
	
	
	
	
	
	
	
	
	
