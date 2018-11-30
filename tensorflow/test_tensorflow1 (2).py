import numpy as np 
import tensorflow as tf 

#create data 
x_data = np.random.rand(100).astype(np.float32)    #在tensorflow中数值类型为float32时更容易被识别，不容易出错
y_data = x_data * 0.1 +0.3

## create tensorflow structure start 
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y= Weights *x_data +biases

loss = tf.reduce_mean(tf.square(y- y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#init =tf.initiallize_all_variables()     #在tensorflow中十分重要的一步
init = tf.global_variables_initializer()
##tensorflow structure end

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step%20 ==0:
		print(step, sess.run(Weights), sess.run(biases))
