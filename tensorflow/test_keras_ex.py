
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
y_test_1 = np_utils.to_categorical(y_test, nb_classes)
print(x_test_1.shape, y_test_1.shape)

'''
#构建模型
model = Sequential()
model.add(Dense(nb_classes, input_shape=(784, )))   #全连接， 输入为784维度， 输出10维度，需要和输入输出对应
model.add(Activation('softmax'))

sgd = SGD(lr = 0.005)
#binary_crossentropy--交叉熵函数
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#model 概要
model.summary()

#训练模型
model.fit(x_train_1, y_train_1, nb_epoch=20, batch_size=100)

#测试模型
loss, accuracy = model.evaluate(x_test_1, y_test_1)
print('test_loss:', loss)
print('test_accuracy:', accuracy)
'''

'''
######构建一个五层的sigmoid 全连接神经网络#######


model = Sequential()
model.add(Dense(200,input_shape=(784, )))  #全连接，输入784维度，输出10维度，需要和输入输出对应
model.add(Activation('sigmoid'))
model.add(Dense(100))  #除了首次需要设置输入维度，其他层只需要输入输出维度就可以了，输入维度会自动继承上层
model.add(Activation('sigmoid'))
model.add(Dense(60))
model.add(Activation('sigmoid'))
model.add(Dense(30))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = Adam(lr=0.003)
model.compile(loss='binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#model 概要
model.summary()

#模型训练
model.fit(x_train_1, y_train_1, nb_epoch=20, batch_size=100)

#模型的测试误差指标
print(model.metrics_names)
#对测试数据进行测试
loss, accuracy = model.evaluate(x_test_1,y_test_1, batch_size=100)
print('test loss:' ,loss)
print('test accuracy: ',accuracy)
###测试结果：test loss: 0.040849     ,test accuracy :0.9859599
'''


'''
###########改变学习率，可能回达到更好的计算结果，学习速率大小的调节一般取决于loss的变化幅度,激活函数改为relu函数
model = Sequential()
model.add(Dense(200, input_shape=(784, ))) #全连接，输入784维度，输出10维度，需要和输入输出对应
model.add(Activation('relu'))   #将激活函数sigmoid 改为Relu
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = Adam(lr=0.001)    #学习速率设置为0.001
model.compile(loss = 'binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#model 概要
model.summary()

#model 训练
model.fit(x_train_1, y_train_1, nb_epoch=30, batch_size=100)

#模型的测试误差指标
loss, accuracy = model.evaluate(x_test_1,y_test_1, batch_size=100)
print('test loss:', loss)
print('test accuracy:', accuracy)
#测试结果：test loss:0.0188449,  test accuracy:0.99569

'''


###添加dropout层，防止模型过拟合，使得网格单元按照一定的概率将其暂时从网络中丢弃，从而解决过拟合问题
model = Sequential()
model.add(Dense(200, input_shape=(784,)))  #全连接，输入784维度，输出10维度，需要和输入输出对应
model.add(Activation('relu'))   #将激活函数sigmoid 改为relu
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))  #添加一个dropout层，随机移除25%的单元
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('softmax'))

##如果是多分类文题，输出层的激活函数一般选用softmax;如果是二分类问题，输出层的激活函数一般选用sigmoid函数

sgd = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model 概要
model.summary()

#训练模型
model.fit(x_train_1, y_train_1, nb_epoch=30, batch_size=100)

#模型的测试误差指标
loss, accuracy = model.evaluate(x_test_1, y_test_1, batch_size=100)
print('test_loss:', loss)
print('test_accuracy', accuracy)
##测试结果：test_loss:0.028454, test_accuracy: 0.9943

