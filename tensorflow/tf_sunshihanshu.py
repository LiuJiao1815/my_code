'''
#经典损失函数
#分类问题中使用的cross_entropy 函数
import tensorflow as tf 
v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
sess = tf.InteractiveSession()
print(tf.clip_by_value(v, 2.5, 4.5).eval()) 

#自定义损失函数，以下代码展示了tf.select函数和tf.greater函数的用法
import tensorflow as tf 
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0,2.0,1.0])

sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())
print(tf.where(tf.greater(v1, v2), v1, v2).eval())
sess.close()
'''
import tensorflow as tf 
from numpy.random import RandomState
batch_size = 8

#两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
#回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape = (None, 1), name='x-input')

#定义了一个单层的神经网络前向传播的过程，这里就是简单地加权和
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x,w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), 
	(y-y_) * loss_more,
	(y_-y)*loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 +x2 +rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

#训练神经网络
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	STEPS = 5000
	for i in range(STEPS):
		start = (i * batch_size) % dataset_size
		end = min(start +batch_size, dataset_size)
		sess.run(train_step,
			feed_dict = {x: X[start:end], y_: Y[start:end]})
		print(sess.run(w1))