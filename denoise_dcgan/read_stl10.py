
import os, sys, tarfile, urllib
import numpy as np
import Image

height = 96
width = 96

data_path = './data/stl10_binary/train_X.bin'

f = open(data_path, 'rb')

everything = np.fromfile(f, dtype=np.uint8)
images = np.reshape(everything, (-1, 3, 96, 96))

images = np.transpose(images, (0, 3, 2, 1))

print images.shape

mean = 0
sigma = 50

for i in range(len(images)):
    new_img = Image.fromarray(images[i], 'RGB')
    new_img.save('./data/stl10_binary/real_images/'+str(i)+'.png')

    gauss = np.random.normal(mean, sigma, (height*width)).reshape(height, width)

    noisy = images[i].astype(np.float32)
    noisy[:,:,0] = noisy[:,:,0] + gauss
    noisy[:,:,1] = noisy[:,:,1] + gauss
    noisy[:,:,2] = noisy[:,:,2] + gauss

    noisy = noisy - np.min(noisy)
    noisy = noisy / np.max(noisy)
    noisy = (noisy*255).astype(np.uint8)
    
    new_img = Image.fromarray(noisy, 'RGB')
    new_img.save('./data/stl10_binary/noise_images/'+str(i)+'.png')


    

    









