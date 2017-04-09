import os
import tensorflow as tf
import pdb

from .layers import *

class CNN():
    def __init__(self, sess,
                 observation_dims,
		 color_channel=3,
		 data_format='NHWC',
                 trainable=True,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=tf.nn.tanh,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.constant_initializer(0.1),
                 value_hidden_sizes=[512],
                 advantage_hidden_sizes=[512],
                 name='CNN'):

	self.sess = sess
	self.name = name
	self.num_action = 3
	self.var = {}


	self.sigma = tf.Variable(0.5, trainable=False)

        self.inputs = tf.placeholder('float32', [None] + observation_dims + [color_channel], name='inputs')
        self.l0 = tf.div(self.inputs, 255.)

        with tf.variable_scope(self.name):
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
                32, [8, 8], [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l1_conv')
            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l2_conv')
            self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l3_conv')

            self.l4, self.var['l4_w'], self.var['l4_b'] = \
                linear(self.l3, 512, weights_initializer, biases_initializer,
                hidden_activation_fn, trainable, name='l4_conv')
            self.outputs, self.var['w_out'], self.var['b_out'] = \
                linear(self.l4, self.num_action, weights_initializer, biases_initializer, 
	        None, trainable, name='out')	# B, A*2
	    
	    self.mu = self.outputs	# B, A	    
	    #self.sigma = tf.nn.sigmoid(self.outputs[:,self.num_action:]) + 1e-5	# B, A
            self.mu_sum = tf.scalar_summary('mu', tf.reduce_mean(self.mu))
            self.sigma_sum = tf.scalar_summary('sigma', tf.reduce_mean(self.sigma))

            self.y1_sum = tf.scalar_summary('y1', tf.reduce_mean(self.mu[:,0]))
            self.y2_sum = tf.scalar_summary('y2', tf.reduce_mean(self.mu[:,1]))
	    self.y3_sum = tf.scalar_summary('y3', tf.reduce_mean(self.mu[:,2]))


	    self.normal_dist = tf.contrib.distributions.Normal(self.mu, tf.stop_gradient(self.sigma))	# B normal distribution
	    self.action = tf.squeeze(self.normal_dist.sample_n(1))

    def calc_action(self, observation):
	return self.action.eval({self.inputs: observation}, session=self.sess) 
   
    
