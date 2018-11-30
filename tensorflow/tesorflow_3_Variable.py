import tensorflow as tf
'''
#声明 w1, w2 两个变量。这里还通过seed参数设定了随机种子
#这样可以保证每次运行得到的结果是一样的

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#暂时将输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])

#通过前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)  #矩阵相乘
y = tf.matmul(a, w2)

sess = tf.Session()
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()    #两条命令的作用是一样的，初始化变量
sess.run(init)
print(sess.run(y))
sess.close
'''
###############################################################
'''
import tensorflow as tf 

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1, seed=1))

#定义placeholder作为存放输入数的地方。这里维度也不一定要定义，如果维度是确定的，那么给出维度可以降低出错的概率

x = tf.placeholder(tf.float32, shape=(None,2), name = 'input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

init = tf.global_variables_initializer()   #初始化变量

with tf.Session() as sess :
	sess.run(init)
	#print(sess.run(y, feed_dict = {x: [[0.7, 0.9]]}))
	print(sess.run(y, feed_dict = {x: [[0.7, 0.9], [0.1, 0.4], [0.5,0.8]] }))
'''
############################################################

#定义一个完整的程序来训练神经网络解决二分类问题
import tensorflow as tf 
from numpy.random import RandomState

#定义训练数据的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

'''
在shape的一个维度上使用None可以方便使用不同的batch大小， 在训练集时需要把数据分为比较小的batch
但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据
放在一个batch可能会导致内存溢出
'''
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

#定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(
	y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个数值模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#定义规则来给出样本的标签，在这里所有x1+x2<1的样例都被认为是正样本，而其他俄日负样本， 1表示正样本，
#0表示负样本
Y = [[int(x1+x2<1)] for (x1, x2) in X]

#创建一个会话来运行TensorFlow 程序
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()     #初始化变量
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))


	#设定训练的轮数
	STEPS = 5000
	for i in range(STEPS):
		#每次选出batch_size个样本进行训练。
		start = (i * batch_size) %dataset_size
		end = min(start+batch_size, dataset_size)

		#通过选取的样本训练神经网络并更新参数
		sess.run(train_step, feed_dict = {x:X[start:end], y_: Y[start:end]})
		if i % 1000 ==0:
			#每隔一段时间计算在所有数据上的交叉熵输出。
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_ :Y})
			print("after %d training step(s), cross entropy on all data is %g" , (i, total_cross_entropy))
	print(sess.run(w1))
	print(sess.run(w2))

'''
训练神经网络的过程可以分为三个步骤：
1. 定义神经网络的结构和前向传播的输出结果
2. 定义损失函数以及选择反向传播优化的算法
3. 生成会话（tf,Session）并且在训练数据上反复运行方向传播优化算法，无论神经网络的结构如何变化，这三个步骤是不变的
'''
