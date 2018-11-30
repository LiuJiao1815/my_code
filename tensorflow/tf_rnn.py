import numpy as np 

X = [1,2]
state = [0.0, 0.0]

#分开定义不同的输入部分的权重以方便操作

w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

#定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_cell = np.asarray([0.1, -0.1])

#定义用于输出的全连接层参数

w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
	#计算循环体中的全连接层神经网络
	before_activation = np.dot(state, w_cell_state) + X[i] *w_cell_input + b_cell

	state = np.tanh(before_activation)


	#根据当前时刻状态计算最终结果
	final_output = np.dot(state, w_output) + b_output

	#输出每个时刻的信息
	print('before activation :'  ,before_activation)
	print('state:', state)
	print('output:', final_output )



#定义一个LSTM结构
#定义一个LSTM结构， 在tensorflow中通过一句简单的命令就可以实现一个完整LSTM结构
#LSTM中使用的变量也会在该函数中自动被声明

lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将LSTM中得状态初始化为全0数组，和其他神经网络类似，在优化循环神经网络时，每次也会使用
#batch 的训练样本，在以下代码中， batch_size给出了一个batch的大小

#BasicLSTMCell 类提供了zero_state 函数来申城全零的初始状态

state = lstm.zero_state(batch_size, tf.float32)

#定义损失函数
loss = 0.0

#考虑到循环神经网络在处理长度太长的序列时会出现梯度消散的问题，因此在训练时为了避免梯度消散的问题
#会规定一个最大的序列长度，在以下的=代码中，用num_steps来表示这个长度

for i in range(num_steps):
	#在第一个时刻声明LSTM结构使用中的变量， 在之后的时刻都需要服用之前定义好的变量

	if i>0:
		tf .get_variable_scope().reuse_variables()

#每一步处理时间序列中的一个时刻，将当输入（current_input) 和之前一时刻状态（state)传入定义的
#LSYM结构的输出lstm_output和更新后的状态state

lstm_output, state = lstm(current_input, state)
#将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出

final_output = fully_connected(lstm_output)

#计算当前时刻输出的损失
loss += calc_loss(final_output, expected_output)