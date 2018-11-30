import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

########two feature input#####
#prepare data
x = np.linspace(-1, 1, 300)
np.random.shuffle(x)
x = x.reshape([100, 3])
y = np.power(x[:,0], 1)+ np.power(x[:, 1], 2) + np.power(x[:,2], 3)
noise = np.random.normal(0, 0.1, y.shape)
y  = y + noise
print(y.shape)




y = y.reshape([-1, 1])
print(y)



#for test 
x1 = np.linspace(-1,1, 300)
np.random.shuffle(x1)
x1 = x1.reshape([100, 3])
y1 =  np.power(x1[:,0], 1) + np.power(x1[:,0],2) +np.power(x1[:, 1],3)
noise = np.random.normal(0, 0.1, y1.shape)
y1 = y1 + noise
y1 = y.reshape([-1, 1])


def add_layer(input_data, input_feature_num, output_feature_num, activation=None):
	Weights = tf.Variable(tf.random_normal([input_feature_num, output_feature_num]))
	biases = tf.Variable(tf.random_normal([1, output_feature_num]) + 0.1)
	Wx_plus_b = tf.matmul(input_data, Weights) + biases
	if activation == None:
		return Wx_plus_b
	else :
		return activation(Wx_plus_b)


tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)

output_1 = add_layer(tf_x, 3, 10, activation=tf.nn.relu)
output_2 = add_layer(output_1, 10, 1, activation=None)
#output_3 = add_layer(output_2, 3, 1 , activation=None)

loss = tf.losses.mean_squared_error(tf_y, output_2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(loss)



##session running------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []

for step in range(20000):
	_, cost, pred = sess.run([train_op, loss, output_2], feed_dict={tf_x: x, tf_y:y})
	if step%10 ==0 and step>1000:
		cost_list.append(cost)
		print(step, cost)

plt.plot(cost_list)
plt.show()
plt.scatter(y, pred, c = 'r')
plt.show()
pred1 = sess.run(output_2, feed_dict={tf_x:x1})



