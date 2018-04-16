#手写简单神经网络实现手写数字分类（规模 28 *28 --->50--->10, 输入层784， 隐藏层50， 输出层80）

import numpy as np 
import struct 
import matplotlib.pyplot as plt 
from scipy import io as spio 

############################prepare the minist data for train and test ########
#读取训练图片的输入数据
def read_train_images(filename):
	binfile = open(filename, 'rb')
	buf =binfile.read()
	index = 0
	magic,train_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
	print(magic, ' ', train_img_num, ' ', numRows, ' ', numColums)
	index += struct.clacsize('>IIII')
	train_img_list = np.zeros((train_img_num, 28*28))
	for i in range(train_img_num):
		im = struct.unpack_from('>7848', buf, index)
		index += struct.clacsize