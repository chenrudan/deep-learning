
import numpy as np
import tensorflow as tf
import time

def xavier_init(fan_in, fan_out, constant=1):
	""" Xavier initialization of network weights"""
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out),
			minval=low, maxval=high,
			dtype=tf.float32)

class Autoencoder(object):
    """Initilize the autoencoder neural network

	Attributes:
        sess: the tensorflow session
	network_architecture: neural number of each layer
	transfer_fct: activation function, default is sigmoid
	lr: learning rate
	batch_size: 
    """
    def __init__(self, sess, network_architecture,
			transfer_fct=tf.nn.sigmoid,
			learning_rate=0.001, batch_size=100):
        """initlize the parameters and construct whole network"""

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.lr = learning_rate
        self.batch_size = batch_size

        print "units number in each layer: " + str(self.network_architecture)
        print "learning rate: " + str(self.lr)
        print "batch size is: " + str(self.batch_size)

	    #input of whole network
        self.x = tf.placeholder(tf.float32, [None, network_architecture[0]])
        self.W = []
        self.bias = []
        
        self._create_network()

        self._create_loss_optimizer()

        init = tf.initialize_all_variables()

        self.sess =sess
        self.sess.run(init)

        self.saver = tf.train.Saver(tf.trainable_variables())

    def _create_network(self):
        """according to the neural number in each layer, initilize 
            weight and bias, then connect the whole net.

        """
        self._create_forward(self.x)
        self._create_backward(self.y)

    def _create_forward(self, x):
        for i in xrange(1, len(self.network_architecture)):
            y = self._create_one_layer(x, self.network_architecture[i], i)
            x = y
        self.y = y

    def _create_backward(self, x):
        for i in xrange(len(self.network_architecture)-2, -1, -1):
            y = self._create_one_layer(x, self.network_architecture[i], i, is_encoder=False)
            x = y
        self.reconstruct_x = y

    def _create_one_layer(self, x, num_out, w_id, is_encoder=True):
        """construct one encoder or decoder layer

        Args:
            x: input of this layer
            num_out: neural number of this layer
            w_id: if this layer is decoder, weight of this layer comes from the saved weight list
            is_encoder: if True, create the new weight variable

        Returns:
            y: output of this layer
         """
        if not is_encoder:
            weight = tf.transpose(self.W[w_id])
        else:
            (batch_size, num_in) = tf.Tensor.get_shape(x).as_list()
            weight = tf.Variable(xavier_init(num_in, num_out))
            self.W += [weight]

        bias = tf.Variable(tf.zeros([num_out], dtype=tf.float32))
        self.bias += [bias]
        y = self.transfer_fct(tf.add(tf.matmul(x, weight), bias))

        return y

    def _create_loss_optimizer(self):
	"""construct the cost function

	reconstruction loss which comes from the cross entropy

	"""
	self.cost = -tf.reduce_sum(self.x * tf.log(1e-10 + self.reconstruct_x)
				+ (1-self.x) * tf.log(1e-10 + 1 - self.reconstruct_x), 1)
	self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def partial_fit(self, X):
	"""training one batch

        Args:
            X: input of this batch

        Returns:
            cost: cost of this batch

	"""
	opt, cost = self.sess.run((self.optimizer, self.cost),
					    feed_dict={self.x: X})
	return cost

    def set_params(self):
        self.saver.restore(self.sess, "model.ckpt")

    def reconstruction_error(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def get_params(self):
        return self.W

    def fit(self, X, num_epochs=100):
        n_samples = len(X)
        total_batch = int(len(X)/self.batch_size)

        t = time.time()
        for epoch in xrange(num_epochs):
            avg_cost = 0.0
            for i in xrange(total_batch):
                batch_x = X[i*self.batch_size:(i+1)*self.batch_size]
                cost = self.partial_fit(batch_x)
                avg_cost += cost
            avg_cost = avg_cost / n_samples

            print "Epoch:" + str(epoch) +  " cost=" + str(np.mean(avg_cost))

            if epoch % 10 is 0:
                print 'until current epoch ' + str(epoch) +' cost: ' + str(time.time()-t) + ' s.'
                
       
        #save the net parameters
        self.saver.save(self.sess, "model.ckpt")

    def fit_transform(self, X, num_epochs=100):
        self.fit(X, num_epochs)
        return transform(X)

    def transform(self, X):
        """transform x
	"""
        return self.sess.run(self.y, feed_dict={self.x: X})



