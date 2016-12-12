
# -*- coding: utf-8 -*
import numpy as np
from sklearn import decomposition, manifold
import pickle
import itertools
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D


file_read = open('input_features.bin', 'rb')
s = file_read.read()
input_features = pickle.loads(s)
file_read.close()

high_dim_input = []
for section_id in input_features:
    high_dim_input.append(input_features[section_id])

high_dim_input = np.array(list(itertools.chain.from_iterable(high_dim_input)))

labels = []
for section_id in input_features:
    for i in range(len(input_features[section_id])):
        labels.append(section_id)
labels = np.array(labels)

'''
进行pca降维
'''
pca = decomposition.PCA(n_components=2)
#isomap = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_input = pca.fit_transform(high_dim_input)

colors = [
        '#FC0E77', '#FC0E77', 
        'turquoise', 'turquoise',
        'turquoise', 'turquoise',
        '#FC0E77', '#FC0E77'
        ]

colors = ['#48A946', '#E55523', '#E5E223', '#23E5DF', '#F70DB4', '#0D77F7','#CD2E7C', '#F70D80']
markers = ['1', '2', '3', '4', '5', '6', '7', '8']

s = []
for color, i, marker in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], markers):
    s.append(plt.scatter(X_input[labels == i, 0], X_input[labels == i, 1],
                 color=color, s=100, marker=r"${}$".format(marker)))
plt.legend((s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]), 
        ('1_1.txt', '1_2.txt', '1_3.txt', '1_4.txt', 
         '2_1.txt', '2_2.txt', '2_3.txt', '2_4.txt'), loc='lower left')
plt.title('1278 vs 3456')
plt.show()
