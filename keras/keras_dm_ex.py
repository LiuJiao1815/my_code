
from IPython.display import SVG
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD, Adam 
#from keras.utils.visualize_util import model_to_dot 
from keras.utils import np_utils
import matplotlib.pyplot as plt 

import tensorflow as tf
import pandas as pd 

#设置随机数种子，保证实验可重复

import numpy as np 
np.random.seed(0)
#设置线程
THREADS_NUM = 20
tf.ConfigProto(intra_op_parallelism_threads = THREADS_NUM)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('原始数据结构：')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#数据变换
#分为10个类别
nb_classes = 10
x_train_1 = x_train.reshape(60000, 784)
y_train_1 = np_utils.to_categorical(y_train, nb_classes)
print('变换后的数据结构')
print(x_train_1.shape, y_train_1.shape)

x_test_1 = x_test.reshape(10000, 784)
y_tets_1 = np.utils.to_categorical(y_test, nb_classes)
print(x_test_1.shape, y_tets_1.shape)

#构建模型
model = Sequential()
model.add(Dense(nb_classes, input_shape=(784, )))   #全连接， 输入为784维度， 输出10维度，需要和输入输出对应
model.add(Activation('softmax'))

sgd = SGD(lr = 0.005)
#binary_crossentropy--交叉熵函数
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#model 概要
model.summary()
