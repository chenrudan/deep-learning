""" Filename: crbm.py """
import numpy as np
import os
import Image   
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

def readFile(file_name):
	f = np.fromfile(file_name, dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	return image
"""	for k in range(0,200):
		b =[]
		for i in range(0,96):
			a = []
			for j in range(0,96):
				a.append([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]])
			b.append(a)
		c = np.asarray(b, dtype = np.uint8)
		img = Image.fromarray(c,'RGB')
		img.save(str(k) +'.png')
"""	

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
	return img_data

class CRBM(object):
	"""convolutional Restricted Boltzmann Machine """
	def _init_(self, input = None, input_channel = None, input_size = None, filter_channel = None \
		filter_size = None, pooling_size = None, W = None, vbias = None, hbias = None,  \
		theano_rng = None):
		
		self.input_channel = input_channel
		self.input_size = input_size
		self.filter_channel = filter_channel
		self.filter_size = filter_size
		self.pooling_size = pooling_size
		self.out_size = input_size - filter_size + 1

		if numpy_rng is None:
			numpy_rng = numpy.random.RandomState(1234)
		
		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		
		if W is None:
			"""初始化权重"""
			initial_W = 0.01*np.random.randn(filter_channel, input_channel, filter_size, filter_size)
			W = theano.shared(value=initial_W, name='W', borrow=True)
		
		if vbias is None:
			vbias = theano.shared(value = numpy.zeros(input_channel, dtype = theano.config.floatX),
									name = 'vbias', borrow = True)
		if hbias is None:
			hbias = theano.shared(value = -0.1*numpy.ones(filter_channel, 
									dtype = theano.config.floatX),
									name = 'hbias', borrow = True)
		
		self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

	def convolutionForward(self, vis):
		conv_out = conv.conv2d(input = vis, filters = self.W, 
							filter_shape= self.filter_size, image_shape = self.input_size)
		return [conv_out, T.nnet.sigmoid(conv_out)]
	
	def sampleHidden(self, v0_sample):
		pre_sigmoid_h1, h1_mean = self.convolutionForward(v0_sample)
		h1_sample =  self.theano_rng.binomial(size=h1_mean.shape,
		 									n=1, p=h1_mean,
		 									dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]
		
	def convolutionBackward(self, hid):
		conv_out = conv.conv2d(input = hid, filters = self.W.T, 
							filter_shape= self.filter_size, image_shape = self.out_size)
		return [conv_out, T.nnet.sigmoid(conv_out)]
	
	def sampleHidden(self, h0_sample):
		pre_sigmoid_v1, v1_mean = self.convolutionForward(h0_sample)
		v1_sample =  self.theano_rng.binomial(size=v1_mean.shape,
		 									n=1, p=v1_mean,
		 									dtype=theano.config.floatX)
		return [pre_sigmoid_v1, v1_mean, v1_sample]
	
	def gibbsHVH(self, h0_sample):
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
				pre_sigmoid_h1, h1_mean, h1_sample]
	
	def maxPooling(conv_out):
		pooled_out = downsample.max_pool_2d(input = conv_out,
											ds = self.pooling_size, ignore_border = True)
		return pooled_out
		
	def updates(self, learning_rate = 0.1):
		#计算postive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
		[pre_sigmoid_nvs, nv_means, nv_samples,
		 pre_sigmoid_nhs, nh_means, nh_samples], updates = \
		 	theano.scan(self.gibbsHVH, \
		 			outputs_info = [None, None, None, None, None, None])
		
		
		
	
	
def test_crbm(learning_rate = 0.1, batch_size = 2):
	#1.得到输入图片
	layer1_input_image = readFile('../../../../crd/deeplearning/data/stl10/train_X.bin')
	layer1_input = []
	#2.白化输入
	for i in range(0,50):
		for j in range(0, 3):
			layer1_input.append(preprocess(layer1_input_image, i, j))
	#3.初始化参数
	n_train_batches = 50/batch_size;
	####################
	#     layer 1      #
	####################
	crbm = CRBM(input = layer1_input, input_channel = 3, input_size = 96, filter_channel = 24 \
		filter_size = 15, pooling_size = 2)
	
	
	
	
	
	
	
if __name__ == "__main__":
	test_crbm()
		
	
	
	
	
	
	
	
	
	
	
	
	
	
