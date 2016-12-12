

# -*- coding: utf-8 -*
import mxnet as mx
import numpy as np
import pickle
import itertools

def get_train_val():

    pos_section = np.array([0,1,6,7])

    file_read = open('input_features.bin', 'rb')
    s = file_read.read()
    input_features = pickle.loads(s)
    file_read.close()

    X = []
    for section_id in input_features:
        X.append(input_features[section_id])

    X = np.array(list(itertools.chain.from_iterable(X)))
    
    Y = []
    #
    for section_id in input_features:
        for i in range(len(input_features[section_id])):
            if section_id in pos_section:
                Y.append(1)
            else:
                Y.append(0)
    Y = np.array(Y)

    idx = np.linspace(0, len(Y)-1, num=len(Y), dtype=np.int)
    np.random.shuffle(idx)
    idx = idx[:87]
    

    train_label = Y[idx]
    train_data = X[idx]
    val_label = np.delete(Y, idx)
    val_data = np.delete(X, idx, axis=0)

    return train_label, train_data, val_label, val_data

train_label, train_data, val_label, val_data = get_train_val()

print 'train_data:', train_data.shape
print 'train_label:', train_label.shape
print 'val_data:', val_data.shape
print 'val_label:', val_label.shape

batch_size = 3
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size)

import logging
logging.getLogger().setLevel(logging.DEBUG)

data = mx.sym.Variable('data')
fc = mx.sym.FullyConnected(data=data, name='fc', num_hidden=2)
lr = mx.sym.SoftmaxOutput(data=fc, name='softmax')

model = mx.model.FeedForward(symbol=lr, num_epoch=100, 
        learning_rate=0.01)

model.fit(X = train_iter, eval_data=val_iter, 
        batch_end_callback = mx.callback.Speedometer(batch_size, 10))
i = 0
j = 0
m = 0
n = 0
for k in range(39):
    if model.predict(val_data)[k].argmax() == 1 and val_label[k] == 1:
        i += 1
    elif model.predict(val_data)[k].argmax() == 0 and val_label[k] == 1:
        j += 1
    elif model.predict(val_data)[k].argmax() == 1 and val_label[k] == 0:
        m += 1
    elif model.predict(val_data)[k].argmax() == 0 and val_label[k] == 0:
        n += 1
print '\tPredict 1\tPredict 0'
print 'True 1\t',i,'\t\t',j
print 'True 0\t',m,'\t\t',n
