
import tensorflow as tf
import os
import time
from glob import glob
from utils import *

def generator(gdata, img_size, batch_size, c_dim, num_filter):
    s2 = img_size/2
    s4 = img_size/4
   
    stddev = 0.0001
    with tf.variable_scope('g_conv1') as scope:
        w = tf.get_variable('w', [4, 4, c_dim, num_filter], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        gconv = tf.nn.conv2d(gdata, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(gconv, biases)
        gconv1 = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('g_conv2') as scope:
        w = tf.get_variable('w', [4, 4, num_filter, num_filter*2], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        gconv = tf.nn.conv2d(gconv1, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter*2], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(gconv, biases)
        gconv2 = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('g_conv3') as scope:
        w = tf.get_variable('w', [4, 4, num_filter*2, num_filter*4], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        gconv = tf.nn.conv2d(gconv2, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter*4], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(gconv, biases)
        gconv3 = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('g_deconv1') as scope:
        w = tf.get_variable('w', [4, 4, num_filter*2, num_filter*4], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(gconv3, w, 
                output_shape=[batch_size, s4, s4, num_filter*2], strides=[1, 2, 2, 1]) 
        biases = tf.get_variable('biases', [num_filter*2], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(deconv, biases)
        deconv1 = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('g_deconv2') as scope:
        w = tf.get_variable('w', [4, 4, num_filter, num_filter*2], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(deconv1, w, 
                output_shape=[batch_size, s2, s2, num_filter], 
                strides=[1, 2, 2, 1]) 
        biases = tf.get_variable('biases', [num_filter], 
                initializer=tf.constant_initializer(0.0))
        deconv2 = tf.nn.bias_add(deconv, biases)

    with tf.variable_scope('g_deconv3') as scope:
        w = tf.get_variable('w', [4, 4, c_dim, num_filter], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(deconv2, w, 
                output_shape=[batch_size, img_size, img_size, c_dim], 
                strides=[1, 2, 2, 1]) 
        biases = tf.get_variable('biases', [c_dim], 
                initializer=tf.constant_initializer(0.0))
        deconv3 = tf.nn.bias_add(deconv, biases)

    return tf.nn.tanh(deconv3)

def discriminator(ddata, batch_size, c_dim, num_filter, leak, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    stddev = 0.0002
    with tf.variable_scope('d_conv1') as scope:
        w = tf.get_variable('w', [4, 4, c_dim, num_filter], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        dconv = tf.nn.conv2d(ddata, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(dconv, biases)
        dconv1 = tf.maximum(bias, leak*bias)

    with tf.variable_scope('d_conv2') as scope:
        w = tf.get_variable('w', [4, 4, num_filter, num_filter*2], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        dconv = tf.nn.conv2d(dconv1, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter*2], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(dconv, biases)
        dconv2 = tf.maximum(bias, leak*bias)

    with tf.variable_scope('d_conv3') as scope:
        w = tf.get_variable('w', [4, 4, num_filter*2, num_filter*4], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        dconv = tf.nn.conv2d(dconv2, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter*4], 
                initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(dconv, biases)
        dconv3 = tf.maximum(bias, leak*bias)

    with tf.variable_scope('d_conv4') as scope:
        w = tf.get_variable('w', [4, 4, num_filter*4, num_filter*8], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        dconv = tf.nn.conv2d(dconv3, w, strides=[1, 2, 2, 1], 
                padding='SAME') 
        biases = tf.get_variable('biases', [num_filter*8], 
                initializer=tf.constant_initializer(0.0))
        dconv4 = tf.maximum(bias, leak*bias)

    with tf.variable_scope('d_local1') as scope:
        local_in = tf.reshape(dconv4, [batch_size, -1])
        shape = local_in.get_shape().as_list()

        w = tf.get_variable('w', [shape[1], 1], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable("biases", [1],
                initializer=tf.constant_initializer(0.0))
        dlocal = tf.matmul(local_in, w) + biases

    return tf.nn.sigmoid(dlocal), dlocal
        
def build_model(img_size, batch_size=100, num_filter=32, c_dim=1, leak=0.1):

    noise_images = tf.placeholder(tf.float32, [batch_size] 
            + [img_size, img_size, c_dim], name='noise_images')
    real_images = tf.placeholder(tf.float32, [batch_size] 
            + [img_size, img_size, c_dim], name='real_images')

    G = generator(noise_images, img_size, batch_size, c_dim, num_filter)
    D, D_logots = discriminator(real_images, batch_size, c_dim, num_filter, leak)
    D_, D_logots_ = discriminator(G, batch_size, c_dim, num_filter, leak, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.zeros_like(D_)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.ones_like(D_)))

    d_loss = d_loss_real + d_loss_fake

    t_vars = tf.trainable_variables()
    
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    return G, g_loss, d_loss, d_vars, g_vars, saver

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Learning rate for adam [0.0002]")
flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
flags.DEFINE_string("dataset", "xxx", "The name of dataset []")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 100, "The size of image to use (will be center cropped) [10.]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")

FLAGS = flags.FLAGS

def read_images(c_dim):
    is_grayscale = (c_dim == 1)
    real_data = glob(os.path.join("./data", FLAGS.dataset, "real_images", "*.png"))
    noise_data = glob(os.path.join("./data", FLAGS.dataset, "noise_images", "*.png"))

    real = [get_image(img_file, FLAGS.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in real_data]
    noise = [get_image(img_file, FLAGS.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in noise_data]

    if is_grayscale:
        reals = np.array(real).astype(np.float32)[:,:,:,None]
        noises = np.array(noise).astype(np.float32)[:,:,:,None]
    else:
        reals = np.array(real).astype(np.float32)
        noises = np.array(noise).astype(np.float32)

    return reals, noises

#def train(sess, G, d_loss, d_vars, g_loss, g_vars, saver, c_dim=1):
def train(sess, img_size, batch_size=100, num_filter=16, c_dim=1, leak=0.2):


    noise_images = tf.placeholder(tf.float32, [batch_size] 
            + [img_size, img_size, c_dim], name='noise_images')
    real_images = tf.placeholder(tf.float32, [batch_size] 
            + [img_size, img_size, c_dim], name='real_images')

    G = generator(noise_images, img_size, batch_size, c_dim, num_filter)
    D, D_logots = discriminator(real_images, batch_size, c_dim, num_filter, leak)
    D_, D_logots_ = discriminator(G, batch_size, c_dim, num_filter, leak, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.zeros_like(D_)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logots_, tf.ones_like(D_)))

    d_loss = d_loss_real + d_loss_fake

    t_vars = tf.trainable_variables()
    
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()


    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(g_loss, var_list=g_vars)

    tf.initialize_all_variables().run()

    start_time = time.time()
    counter = 0

    reals, noises = read_images(c_dim)

    sample_images = reals[0:batch_size]
    sample_z = noises[0:batch_size]

    model_name = "DCGAN.model"
    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.image_size)
    checkpoint_dir = os.path.join('./checkpoint', model_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(FLAGS.epoch):

        data = glob(os.path.join("./data", FLAGS.dataset, "real_images", "*.png"))
        num_batch = len(data) // FLAGS.batch_size

        print 'num_batch', num_batch

        for idx in range(0, num_batch):
            
            batch_images = reals[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            batch_z = noises[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            
            #update 
            out1 = sess.run([d_optim], feed_dict={real_images: batch_images, noise_images: batch_z})

            #update G
            out2 = sess.run([g_optim], feed_dict={noise_images:batch_z})

            errD_fake = d_loss_fake.eval({noise_images: batch_z})
            errD_real = d_loss_real.eval({real_images:batch_images})
            errG = g_loss.eval({noise_images: batch_z})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, 
                    idx, num_batch, time.time() - start_time, 
                    errD_fake+errD_real, errG))

            if np.mod(counter, 100) == 1:
                samples, loss1, loss2 = sess.run([G, d_loss, 
                        g_loss], feed_dict={noise_images: sample_z,
                        real_images: sample_images})
                save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format('./sample', epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (loss1, loss2))

            if np.mod(counter, 500) == 2:
                saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)
                
#G, g_loss, d_loss, d_vars, g_vars, saver = build_model(FLAGS.image_size, FLAGS.batch_size)
with tf.Session() as sess:
    #train(sess, G, d_loss, d_vars, g_loss, g_vars, saver)
    train(sess, FLAGS.image_size, FLAGS.batch_size, c_dim=FLAGS.c_dim)
    





