from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense 
from keras.optimizers import Adam 
import numpy as np 
'''
模型说明： 用RNN模型来做mnist数据集的分类问题
因为每个图片的数据是28*28
所以讲模型RNN神经网络的输入设为28
也就是说每一个时刻的输入的数据都是28个， 也就是把图片的每一行28个像素作为每一个时刻RNN的输入，也就是INPUT_SIZE
另外RNN需要设定一个循环的长度，也就是TIME_SIZE，那么把循环次数设定为28， 刚好是一张图片的像素行数
也就是说RNN循环28次后刚好学习一张图片的所有像素数据
另外批量处理每次处理50张
'''
#一些参数设置RNN
TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
OUTPUT_SIZE = 10
LR =0.001 
BATCH_INDEX = 0

#下载数据
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#数据处理
x_train = x_train.reshape((-1, 28, 28))/225.0
x_test = x_test.reshape((-1, 28,28))/225.0

y_train = np_utils.to_categorical(y_train, nb_classes = 10)
y_test = np_utils.to_categorical(y_test, nb_classes = 10)

#创建模型
model = Sequential()

#RNN cell  如果使用tensorflow 作为backend, 必须把None放到batch_size中，否则会报错
model.add(SimpleRNN(batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), output_dim = CELL_SIZE))