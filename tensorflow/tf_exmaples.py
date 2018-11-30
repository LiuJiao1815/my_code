
#1使用梯度下降优化算法
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#定义层函数
def add_layer(inputs, in_size, out_size, activation_function = None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b

	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

#构造数据

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
predition = add_layer(l1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -predition), 
	reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#当有变量时，这是十分重要的一步
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()    #保证图连续输出，保证程序不会因为画一张图而终止
plt.show()


for i in range(1000):
	sess.run(train_step, feed_dict = {xs: x_data, ys:y_data})
	if i%50 ==0:

		try:
			ax.lines.remove(lines[0])   #删除前面的曲线
		except Exception:
			pass
		print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
		predition_value = sess.run(predition, feed_dict ={xs: x_data})
		lines = ax.plot(x_data,predition_value, 'r-', lw = 5)
		plt.pause(0.1)   #暂停0.1秒
# 最小损失为0.00384




'''
#2、使用adam优化算法

import tensorflow as tf 
import numpy as np 

#定义层函数
def add_layer(inputs, in_size, out_size, activation_function = None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b

	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

#构造数据
#batch_size = 10   #每次训练数据样本的个数

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
predition = add_layer(l1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -predition), 
	reduction_indices = [1]))

train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict = {xs: x_data, ys:y_data})
	if i%50 ==0:
		print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))

#优化后最小损失函数为0.00289
'''
