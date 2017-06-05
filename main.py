import logging
import tensorflow as tf
from utils import get_model_dir
from networks.cnn import CNN
from policy import Policy
from environment import Curve
from data_loader import data_loader

flags = tf.app.flags
# Training
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')
flags.DEFINE_string('observation_dims', '[80, 80]', 'The dimension of gym observation')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_float('learning_rate_decay_step', 10000, 'The learning rate of training (*= scale)')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
tf.app.flags.DEFINE_integer('max_iter', 15000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('max_to_keep', 20, 'Max number of model to save.')
tf.app.flags.DEFINE_string('ckpt_dir', './checkpoint', 'Where should we save the model')
tf.app.flags.DEFINE_string('ckpt_path', './checkpoint/checkpoint-5001', 'the model will be restored')
tf.app.flags.DEFINE_string('test_image_path', './data/test/test1.jpg', 'the image we are going to process')
# Data
tf.app.flags.DEFINE_string('source_path', './data/img_20000', 'the model will be restored')
tf.app.flags.DEFINE_string('target_path', './data/img_20000_contrast', 'the image we are going to process')
# Optimizer
flags.DEFINE_float('learning_rate', 0.00025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 0.00025, 'The minimum learning rate of training')
flags.DEFINE_float('learning_rate_decay', 0.96, 'The decay of learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer')
# Debug
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('gpu_fraction', '1/3', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print (" [*] GPU : %.4f" % fraction)
  return fraction

conf = flags.FLAGS

def main(_):
  # preprocess
  conf.observation_dims = eval(conf.observation_dims)

  # start
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(conf.gpu_fraction))

  dataset = data_loader(conf.source_path, conf.target_path)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    env = Curve()

    pred_network = CNN(sess=sess,
                       observation_dims=conf.observation_dims,
                       name='pred_network', 
		       trainable=True)

    policy = Policy(sess=sess, 
		    pred_network=pred_network,
		    env=env,
		    dataset=dataset,
		    conf=conf)

    if conf.is_train:
        policy.train()
    else:
	policy.test(conf.test_image_path)

if __name__ == '__main__':
  tf.app.run()
