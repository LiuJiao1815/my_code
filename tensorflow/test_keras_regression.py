'''
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 


#create some data for x, y .....
x = np.linspace(-1,1,100)
np.random.shuffle(x)
y = 0.5 *x - 0.35 +np.random.normal(0, 0.05, (100, ))
print(y)
plt.scatter(x,y)
#plt.show()

# get the train and test data------
x_train, y_train = x[:70], y[:70]
x_test, y_test = x[70:], y[70:]

#build the model 
model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))
model.compile(loss='mse', optimizer ='sgd')

#training--------
print('trianing......')
cost_list=[]
for step in range(500):
	cost = model.train_on_batch(x_train, y_train)
	if step %5 ==0:
		print('training cost:',cost)
		cost_list.append(cost)
plt.plot(cost_list)
#plt.show()


#testing--------
print('testing......')
cost = model.evaluate(x_test, y_test)
print('test cost', cost)


'''


'''
#以下是拟合一个二次函数的keras 模型
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


#create some data for x, y 
x = np.linspace(-1, 1,500)
np.random.shuffle(x)
y = 0.5 *x*x - 0.35*x +1.1 +np.random.normal(0,0.05,(500, ))

#get the train and test data 
x_train, y_train = x[:400], y[:400]
x_test, y_test = x[400:], y[400:]

#build the model-----
model = Sequential()
model.add(Dense(output_dim=2, input_dim=1))
model.add(Dense(output_dim=5, input_dim = 2, activation = 'relu'))
model.add(Dense(output_dim=1, input_dim = 5, activation = 'relu'))
model.compile(loss = 'mse', optimizer = 'sgd')

#traing----
print('training....')
cost_list = []
for step in range(2000):
	cost = model.train_on_batch(x_train, y_train)
	if step %100 ==0:
		print('training cost:', cost)
		cost_list.append(cost)

plt.plot(cost_list)
plt.show()


#testing-------------------------------
print('testing......')

cost = model.evaluate(x_test, y_test)
print('test cost', cost)

w,b = model.layers[0].get_weights()

print('weights is :', w, 'bias is:', b)


#ploting the result of test -------------
y_pred = model.predict(x_test)
plt.scatter(x_test, y_pred, c = 'r')
plt.scatter(x_test, y_test, c= 'b')

plt.show()
'''

'''
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 


#create some data for x,y---------------------
x = np.linspace(-1,1, 300)
np.random.shuffle(x)
y = 0.5*x**3 - 0.35*x +np.random.normal(0, 0.03, (300, ))
#plt.scatter(x,y)
#plt.show()

#获取训练数据和测试数据
x_train = x[:250]
y_train = y[:250]
x_test = x[250:]
y_test = y[250:]


#构建神经网络模型
model = Sequential()
model.add(Dense(output_dim =5, input_dim=1,activation = 'relu'))
model.add(Dense(output_dim =8, input_dim=5, activation='relu'))
model.add(Dense(output_dim =20, input_dim=8, activation = 'relu'))
model.add(Dense(output_dim =4, input_dim=20, activation = 'relu'))
model.add(Dense(output_dim =1, activation = 'linear'))

model.compile(loss='mse', optimizer='sgd')

#模型概要
model.summary()

#训练模型
model.fit(x_train,y_train, nb_epoch=2, batch_size=10)

#测试损失
cost = model.evaluate(x_test, y_test)
print('test cost:', cost)

w1,b1 = model.layers[0].get_weights()
w2,b2 = model.layers[1].get_weights()
w3,b3 = model.layers[2].get_weights()
w4,b4 = model.layers[3].get_weights()
w5,b5 = model.layers[4].get_weights()

print('w1, b1:',(w1, b1))
print('w2, b2:',(w2, b2))
print('w3, b3:',(w3, b3))
print('w4, b4:',(w4, b4))
print('w5, b5:',(w5, b5))
'''


import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt 

#构造数据
x = np.linspace(-2,2,5000)
np.random.shuffle(x)
y = 0.5*x**4+ 0.3*x**3+0.2*x**2- 0.05*x + np.random.normal(0, 0.1, (5000, ))
plt.scatter(x,y)
plt.show()


#构造训练数据和测试数据
x_train, y_train = x[:400], y[:400]
x_test, y_test = x[400:], y[400:]

#构建神经网络模型
model = Sequential()
model.add(Dense(output_dim=5, input_dim=1, activation='relu'))
model.add(Dense(output_dim=8, input_dim =5, activation = 'relu'))
model.add(Dense(output_dim=20, input_dim=8, activation ='relu'))
model.add(Dense(output_dim=30, input_dim=20, activation = 'relu'))
model.add(Dense(output_dim=4, input_dim=30, activation = 'relu'))
model.add(Dense(output_dim=1, input_dim=4))

model.compile(loss='mse', optimizer='sgd')
model.fit(x_train,y_train, nb_epoch=20, batch_size=20)
train_loss = model.evaluate(x_train,y_train)
test_loss = model.evaluate(x_test, y_test)
print('train_loss：', train_loss)
print('test_loss：', test_loss)

