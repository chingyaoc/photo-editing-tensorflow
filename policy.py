import tensorflow as tf
import numpy as np
import math
import os, sys
from utils import *
from ops import *
import time
from tqdm import tqdm
import pdb

class Policy():
    def __init__(self, sess, pred_network, env, dataset, conf):
	self.sess = sess
        self.pred_network = pred_network
        self.env = env
	self.dataset = dataset

	self.ckpt_dir = conf.ckpt_dir 
	self.ckpt_path = conf.ckpt_path
	self.max_iter = conf.max_iter 
	self.max_to_keep = conf.max_to_keep
	self.batch_size = conf.batch_size
	self.df_dim = 64
	
        self.learning_rate = conf.learning_rate
        self.learning_rate_minimum = conf.learning_rate_minimum
        self.learning_rate_decay = conf.learning_rate_decay
        self.learning_rate_decay_step = conf.learning_rate_decay_step

	self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.build_opt()

    def build_opt(self):	
	with tf.variable_scope('optimizer'):
            '''Discriminator'''
            self.images = tf.placeholder(tf.float32, [self.batch_size] + [80, 80, 3], name='real_images')
            self.images_ = tf.placeholder(tf.float32, [self.batch_size] + [80, 80, 3], name='fake_images')
            self.D, self.D_logits = self.discriminator(self.images)
            self.D_, self.D_logits_ = self.discriminator(self.images_, reuse=True)

            ones_soft = tf.random_uniform(tf.shape(self.D), 0.8, 1.1)
            zeros_soft = tf.random_uniform(tf.shape(self.D_), 0.0, 0.3)

            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, ones_soft))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, zeros_soft))
            #self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            #self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

            self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'd_' in var.name]
            self.d_optimizer = tf.train.AdamOptimizer(0.0002).minimize(self.d_loss, var_list=self.d_vars)


	    '''Policy'''
	    self.action = tf.placeholder('float32', [None, 3], name='action')    
            self.observation = tf.placeholder(tf.float32, [self.batch_size] + [80, 80, 3], name='observation')
            # Acquire perceptual reward
            self.target, _ = self.discriminator(self.observation, reuse=True)
	    self.log_p = self.pred_network.normal_dist.log_prob(self.action)
	    # Loss and train op
	    self.p_loss = tf.reduce_mean(tf.mul(-tf.reduce_sum(self.log_p,1), tf.mul(self.target, 2))) 
            # Add cross entropy cost to encourage exploration
            #self.p_loss -= 1e-2 * self.pred_network.normal_dist.entropy()
	    # L1 regulization on action
	    self.p_loss += 0.01*tf.reduce_mean(tf.abs(self.action))

            self.action_sum = tf.scalar_summary("action", tf.reduce_mean(self.action))
            self.target_sum = tf.scalar_summary("reward", tf.reduce_mean(self.target))
            self.p_loss_sum = tf.scalar_summary("p_loss", self.p_loss)

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.global_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))

            t_vars = tf.trainable_variables()
            self.p_vars = [var for var in t_vars if 'pred_network' in var.name]
            self.p_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_op).minimize(self.p_loss, var_list=self.p_vars, global_step=self.global_step)

    def predict(self, state):
	action = self.pred_network.calc_action(state)
	return action

    def update(self, state, observation, action, image_real, idx):
        #Update Policy
	for i in range(1):
           feed_dict = {self.pred_network.inputs:state, self.observation:observation, self.action:action}
           _, p_loss, p_summary_str, reward = self.sess.run([self.p_optimizer, self.p_loss, self.p_sum, tf.reduce_mean(self.target)], feed_dict)

        # Update Discriminator
	for i in range(1):
            feed_dict = {self.images:image_real, self.images_:observation}
            _, d_loss, d_summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_sum], feed_dict)

        return p_loss, d_loss, p_summary_str, d_summary_str, reward

    def train(self):
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

        self.p_sum = merge_summary([self.action_sum, self.target_sum, self.p_loss_sum, self.pred_network.mu_sum, self.pred_network.sigma_sum, self.pred_network.y1_sum, self.pred_network.y2_sum, self.pred_network.y3_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)
        tf.initialize_all_variables().run()

        print ' [*] PRE-TRAIN D'
        for idx in range(500):
            image, image_real, pr= self.dataset.sequential_sample(self.batch_size)

            feed_dict = {self.images:image_real, self.images_:image}
            _, d_loss = self.sess.run([self.d_optimizer, self.d_loss], feed_dict)

	print ' [*] START TRAINING'
	for idx in range(self.max_iter):	
            image, image_real, pr= self.dataset.sequential_sample(self.batch_size)

            tSTART = time.time()
            # 1. predict
            action = self.predict(image)
	    # Quantize the action [TODO]
	    action = np.around(action)

	    # 2. act
	    observation = self.env.step(action, pr, self.batch_size)

	    # 3. update
            p_loss, d_loss, p_summary_str, d_summary_str, reward = self.update(image, observation, action, image_real, idx)
            self.writer.add_summary(p_summary_str, idx)
	    self.writer.add_summary(d_summary_str, idx)
	

            tEND = time.time()
            elapse = round(tEND - tSTART,2)


            if idx%500 == 0:
                print ' [*] Iteration', idx,'p_loss:',p_loss,'d_loss:',d_loss, 'action', np.mean(action), 'reward', reward	        
	        self.test('./data/test/test1.jpg', idx)
	        #self.test('./data/test/test2.jpg', idx)
		
            if idx%5000 == 0:
                self.save_model()


    def test(self, img_path, tag):
	#self.load_model(self.ckpt_path)
	image, image_pr = self.dataset.load_image(img_path)
	
	# 1. predict
	action = self.predict(image)
	# 2. act
	observation = self.env.step_test(action, image_pr)

	self.dataset.save_image(observation, tag)	

    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            image_norm = tf.div(image, 255.)
            h0 = lrelu(conv2d(image_norm, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            # Prevent NAN
            h4 += 1e-5
            h4_ = tf.nn.sigmoid(h4) + 1e-5

            return h4_, h4


    def save_model(self, name='checkpoint'):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        print ' [$] Saving model', name+'-'+str(int(self.global_step.eval()))
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, name), global_step=int(self.global_step.eval()))

    def load_model(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            error_msg = "Can't find the checkpoint %s" % ckpt_path
            sys.exit(error_msg)
        else:
            try:
                self.saver.restore(self.sess, ckpt_path)
            except AttributeError:
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                self.saver.restore(self.sess, ckpt_path)
	    
