
# -*- coding: utf-8 -*
import codecs
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt


file_read = open('input_features.bin', 'rb')
s = file_read.read()
input_features = pickle.loads(s)
file_read.close()

X = []
num_1 = 0
for section_id in input_features:
    if section_id < 4:
        num_1 += len(input_features[section_id])
    X.append(input_features[section_id])

Y = np.array(list(itertools.chain.from_iterable(X)))

idx = np.linspace(0, len(Y[0])-1, num=len(Y[0]), dtype=np.int)
np.random.shuffle(Y[num_1:])

print num_1

for i in range(10):
    plt.plot(idx, Y[i+num_1])
plt.xlabel('Feature ID')
plt.ylabel('Feature Count')
plt.title('5~8 Feature Appearence Frequency')
plt.show()

