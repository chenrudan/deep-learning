
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import autoencoder


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

sess = tf.Session()

np.random.seed(0)
tf.set_random_seed(0)

batch_size=100
num_epochs = 50 
display_step=2
network_architecture = [784, 500, 500, 2]

ae = autoencoder.Autoencoder(sess, network_architecture, batch_size=batch_size)
ae.fit(mnist.train.images, num_epochs)                                          

x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = ae.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1)) 
plt.colorbar()
plt.savefig('test_ae.png')



