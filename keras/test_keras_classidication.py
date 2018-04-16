from keras.models import Sequential 
from keras.utils import np_utils
from keras.layers import Dense, Activation 
from keras.optimizers import RMSprop 
import numpy as np 
from keras.datasets import mnist

#create the train data and test data -----
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#data pre_processning 
x_train = x_train.reshape (x_train.shape[0],-1)/225

x_test = x_test.reshape(x_test.shape[0], -1)/225

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)

#building the model--------
model = Sequential([
	Dense(32, input_dim =784),
	Activation('relu'),
	Dense(10),
	Activation('softmax')])

#define optimizer
rmseprop = RMSprop(lr=0.01, rho=0.9, epsilon = 1e-8, decay=0.0)
model.compile(optimizer = rmseprop, loss='categorical_crossentropy', metrics=['accuracy'],)

#training models
model.fit(x_train, y_train, nb_epoch=2, batch_size = 32)

#testing---------t
loss, accuracy = model.evaluate(x_test, y_test)
print('test_loss:', loss)
print('test_accuracy:', accuracy)
##测试结果： test_loss = 0.246687 ; test_accuracy = 0.9462
