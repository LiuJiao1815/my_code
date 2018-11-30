
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

'''
通过tf.get_variable 的方式创建过滤器的权重变量和偏置项变量。上面介绍了卷积层的参数个数只和
过滤器的尺寸、深度以及当前曾节点矩阵的深度油管，所以这里声明的参数变量是一个四维矩阵，前面
两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四学维度表示过滤器的深度
'''
filter_weight = tf.get_variable(
	'weights', [5,5,3,16],
	initializer = tf.truncated_normal_initializer(stddev = 0.1))

#和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不同的偏置项
#本样例代码中16为过滤器的深度，也是神经网络中下一层节点矩阵的深度

biases = tf.get_variable(
	'biases', [16], initializer = tf.constant_initializer(0.1))

'''
tf.nn.conv2d 提供了一个非常方便的函数来实现卷积层前向传播的算法。这个函数的第一个输入为当前层的
节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一个维对应一个输入batch.
比如在输入层，input[0, :, :, :]表示第一张图片，以此类推。tf.nn.conv2d第二个参数提供了卷积层的权重，
第三个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为4 的数组，但是第一维和最后一维的数字
要求一定是1.这是因为卷积层的步长只对矩阵的长和宽有效。最后一个参数是填充的方法， tensorflow中提供
SAME和VALID两种选择。其中SAME表示添加全0填充，VALID表示不添加
'''
conv = tf.nn.conv2d(
	input, filter_weight,strides=[1,1,1,1], padding = 'SAME')

'''
tf.nn.bias_add提供了一个方便的函数给每个节点加上一个偏置项。注意这里不能直接使用加法，因为矩阵
上不同位置上的节点都需要加上相同的偏置项
'''
bias = tf.nn.bias_add(conv, biases)
#将计算的结果通过RELU激活函数完成去线性化
actived_conv = tf.nn.relu(bias)
