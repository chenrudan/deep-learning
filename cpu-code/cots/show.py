""" Filename: show.py """
import numpy as np
import os
import Image   
import theano
import theano.tensor as T

def read_file():
    f = np.fromarray('a.bin', dtype = np.float)
    image = np.array(f, dtype = np.float).reshape((50, 3, 96*96))
    for k in range(0,200):
        b =[]
        for i in range(0,96):
            a = []
            for j in range(0,96):
                a.append([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]])
            b.append(a)
        c = np.asarray(b, dtype = np.uint8)
        img = Image.fromarray(c,'RGB')
        img.save(str(k) +'.png')

if __name__ == "__main__":
    read_file()
