#mnist数据集试验

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_NODE = 784   #输入层的节点数，对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10   #输出层的节点数，这个等于类别的数目，因为在MNIST数据集中，需要区别0-9这十个数字

#配置神经网络参数
LAYER1_NODE = 500 #隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例，隐藏层的节点数为500
BATCH_SIZE = 100  #一个训练batch中的训练数据个数， 数字越小，训练过程越接近随机梯度下降；数值越大越接近梯度下降
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99    #学习率的衰减率
REGULARIZATION_RATE = 0.0001   #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000         #训练轮数
MOVING_AVERAGE_DECAY = 0.99    #滑动平均衰减率


#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果，在这里定义了一个使用RELU激活函数的
#三层全连接神经网络。通过加入隐藏层实现了多层网络结构，通过Relu激活函数实现了去线性化，在这个函数中也支持
#传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型

def inference(input_tensor, avg_class, weights1, biases1, weights1, biases1):
	#当没有提供滑动平均类时，直接使用参数当前的取值
	if avg_class == None:
		#计算隐藏层的前向传播结果，这里使用了Relu激活函数
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

		#计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，
		#所以这里不需要加入激活函数，而且不加入softmax不会影响预测结果，因为预测时
		#使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果的计算
		#没有影响

		return tf.matmul(layer1, weights2) + biases2

	else :
		#首先使用avg_class.average  函数计算得出变量的滑动平均值
		#然后计算响应的神经网络前向传播结果
		layer1 = tf.nn.relu(
			tf.matmul(input_tensor, avg_class.average(weights1)) + 
			avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2)) +
			avg_class.average(biases2)


#训练模型的过程
def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')

	#生成隐藏层参数
	weights1 = tf.Variable(
		tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))

	#生成输出层的参数
	weights2 = tf.Variable(
		tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

	#计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为None,
	#所以函数不会使用参数的活动平均值

	y = inference(x, None, weights1, biases1, weights2, biases2)
	

