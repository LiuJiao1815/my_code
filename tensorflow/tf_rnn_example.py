'''
数据分析: 每一张图片的数据是28*28
所以将每一个图片的每一行看成是一个时序数据，也就是每一张图片表示成28个时序数据
也就是说RNN的输入每一个数据的特征个数是28， input_size = 28
每一个batch 里面包含128个图片的时序数据
也就是一个批次里面的数据是： 128*28*28
'''
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

#数据
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#超参数

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28      #输入的特征数
n_steps = 28       #一共的时刻数
n_hidden_unis = 128     #隐藏神经元的个数

n_classes = 10    #数字为0-9，所以为10类

#定义placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


#定义weights 和 biases 

weights = {
	#(28, 128)
	'in':  tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
	#(128, 10)
	'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}

biases = {
	#(128, )
	'in' : tf.Variable(tf.constant(0.1, shape = [n_hidden_unis, ])),
	#(10, )
	'out': tf.Variable(tf.constant(0.1, shape = [n_classes, ]))

}


#定义RNN， RNN里面的流程是：
#原始数据-->weights['in'], biases['in'] --> cell -->weights['out'], biases['out'] -->out

def RNN(x, weights, biases):

	#-------------------------input ====> hidden layer------------------------
	#x_size ==> [128batch, 28time steps, 28features ] ====>reshape to [128*28, 28]

	x = tf.reshape(x, [-1, n_inputs])

	x_in = tf.matmul(x, weights['in'] + biases['in'])

	#reshape x_in to new shape ---> [128 batch , 28time, 28hidden feature ]
	x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_unis])


	#-----------------------------------hidden layer ===> cell-------------------

	#lstm 中的state 分成两个部分， 一个是主线的state, 另一个是分线的state
	#lstm cell is divided into two parts ==>(core_state, m_state), 即一个元组，里面包含主线的
	#state和分线的state

	#RNN的cell 中只有分线的state
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple = True)
	_init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)

	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state = _init_state, time_major = False)

	#outputs 是包含每一个时刻step的结果， states是最后一个时刻计算获得的状态，dynamic_rnn是一种比较好的计算方式
	#time_najor 是指time_step 这个index是不是在输入矩阵的第一个维度，是的话就是True, 不是的话就是False
	#这里outputs的shape ==> [128batch, 28 time, hidden_units]


	#-----------------------------------cell ==> output layer----------------------

	results = tf.matmul(states[1], weights['out']) +biases['out']    # state[1] ->m_state

	return results

#构造一些需要的计算图
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 0
	while step * batch_size < training_iters:
		batch_xs , batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = tf.reshape(batch_xs,[batch_size, n_steps, n_inputs]) 
		sess.run(train_op, feed_dict = {x: batch_xs, y:batch_ys})
		if step % 20 ==0:
			print(sess.run(accuracy, feed_dict = {x: batch_xs, y: batch_ys}))

		step += 1
'''



'''
数据分析：
每一张图片的数据是 28*28 
所以将每一个图片的每一行看成是一个时序数据，也就是每一张图片表示成28个时序数据
也就是说RNN的输入每一个数据的特征个数是28，input_size = 28
每一个batch里面包含128个图片的时序数据
也就是一个批次里面的数据是： 128 * 28 * 28
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# data 
mnist = input_data.read_data_sets('MNIST_data',one_hot = True )
# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

num_input_fea = 28  #输入特征数
time_steps = 28        #lstm的记忆长度，一共的时刻数
hidden_units = 50     #RNN里面的隐藏神经元数量
batch_size = 128      #(every batch has 128 images matrix)
n_class = 10  
# plaeholder   
x = tf.placeholder(tf.float32,[None , time_steps , num_input_fea])
y = tf.placeholder(tf.float32,[None,n_class])


# define weights
weights={'in': tf.Variable(tf.random_normal([ num_input_fea ,  hidden_units ])) , 'out':tf.Variable(tf.random_normal([ hidden_units, n_class ]))}
biases = {'in':tf.Variable(tf.constant(0.1,shape = [hidden_units,])) , 'out':tf.Variable(tf.constant( 0.1, shape = [n_class , ] ))}


# define RNN   
# RNN网络里的流程：
#原是数据-->weights['in'],biase['in']-->cell-->weights['out'],biases['out']-->output
def RNN(X , weights , biases):
	# ------------------------------input=====> hidden layer ------------------------------------------------------------

	# X_size ==> [128bitach , 28 time steps ,28 features ] ----> reshape to [128*28 , 28]
	X=tf.reshape(X,[-1,num_input_fea])

	# make a matmul ,and have X_in ,shape--> [128batch * 28 times , hidden units num features]
	X_in = tf.matmul(X,weights['in']) + biases['in']

	# reshape x_in to new shape --> [128batch ,28 time , hidden units num features]
	X_in = tf.reshape(X_in,[-1 , time_steps , hidden_units])
               
	#--------------------------------hidden layer ===> cell -----------------------------------------------------

	# lstm 中的state分成了两个部分，一个是主线的state，另一个是分线的state
	# lstm cell is divided into two parts==>( core_state,m_state ),即一个元组，里面包含主线的元组以及分线的元组
	# RNN 的cell 中只有分线的state
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( hidden_units , forget_bias=1.0 , state_is_tuple = True )
	_init_sate = lstm_cell.zero_state(batch_size,dtype=tf.float32)

	outputs , states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_sate , time_major = False)
	#outputs 是包含每一个时刻step的结果 ， states是最后一个时刻计算获得的状态 ，dynamic_rnn是一种比较好的计算形式
	# time_major 是指time_step 这个index是不是在输入矩阵的第一个维度（主要维度），是的话就是True，否则是False
	# 这里outputs的shape ==> [128 batch , 28 time , hidden_units ]

	# -------------------------------cell ==> output layer ---------------------------------------------
	#将output进行格式转换，变成格式===> [(batch_size， outputs ), ...] * TIME STEPS
	results = tf.matmul(states[1], weights['out']) +biases['out']    # state[1] ->m_state
	#outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
	#result = tf.matmul(outputs[-1] , weights['out']) + biases['out'] 
# 利用解开后的outputs 的最后一个元素与weights['out']运算，最后一个就是最后一个时刻的lstm计算结果
	return results

# 构造一些训练需要的计算图
pred =RNN(x,weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))

init = tf.global_variables_initializer()

# session and run
with tf.Session() as sess:
	sess.run(init)
	step = 0
	while step * batch_size < training_iters:
		batch_xs , batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = tf.reshape(batch_xs ,[batch_size , time_steps ,num_input_fea])
		sess.run(train_op ,{x:batch_xs , y:batch_ys})

		if step % 20 ==0:
			print(sess.run(accuracy,{x:batch_xs , y:batch_ys}))
		step =step + 1