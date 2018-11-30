#2、使用adam优化算法

import tensorflow as tf 
import numpy as np 

#定义层函数
def add_layer(inputs, in_size, out_size, activation_function = None):
	#add one more layer return the output of this layer

	with tf.name_scope('layer'): 
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
		with tf.name_scope('inputs'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b

	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

# define placeholder for inputs to network

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

#add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)

#add output layer
predition = add_layer(l1, 10, 1, activation_function = None)

# the error between prediction and real data

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -predition), 
	reduction_indices = [1]), name='loss')

with tf.name_scope('train'):

	train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

#生成一个写日志的writer, 并将当前的tensorflow 计算图写入日志
writer  = tf.summary.FileWriter('C:\\Users\\Administrator\\Desktop\\logs', tf.get_defaut_graph())
sess.run(init)

