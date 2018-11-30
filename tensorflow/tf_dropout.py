import tensorflow as tf 
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt 


#load data from sklearn------------------------
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)         #相当于one_hot转码
print(y)
#prepare train data and test data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  #随机划分训练数据和测试数据

#define function called 'add_layer'---------

def add_layer(inputs, input_fea_num, output_fea_num, layer_name, activation=None):

	Weights = tf.Variable(tf.random_normal([input_fea_num, output_fea_num]))
	biase = tf.Variable(tf.zeros([1, output_fea_num]) + 0.1)   #1行
	Wx_plus_b = tf.matmul(inputs, Weights) + biase

	#添加dropout层， 通过tf.nn.dropout设置keep_pro来实现神经元的dropout功能
	Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

	if activation == None:
		output = Wx_plus_b
	else:
		output = activation(Wx_plus_b)

	#tf.histogram_summary(layer_name + '/outputs', output)

	return output

#define placeholder--------------------
keep_prob = tf.placeholder(tf.float32)             #给dropout的比例定义一个placeholder

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

#add layer-------------------
output1 = add_layer(xs, 64, 100, 'layer1', activation = tf.nn.tanh)
pre = add_layer(output1, 100, 10, 'layer2', activation = tf.nn.softmax)

#loss-----------------------------
cross_entropy  = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pre), reduction_indices = [1]))

#tf.scalar_summary('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#session ----------
sess = tf.Session()

sess.run(tf.initialize_all_variables())
#merged = tf.merge_all_summaries()
#train_writer = tf.train.SummaryWriter('F:\\log\\train', sess.graph)
#test_writer = tf.train.SummaryWriter('F:\\log\\test', sess.graph)

#RUN----------------------
#init = tf.global_variables_initializer()
#sess.run(init)
loss_train = []
loss_test = []

for step in range(500):
	sess.run(train_step, feed_dict = {xs:x_train, ys: y_train, keep_prob: 0.5})

	#keep_prob表示保留下来的神经元的比例

	if step % 10 ==0:
		cost_train = sess.run(cross_entropy, {xs:x_train, ys: y_train, keep_prob: 1.0})
		cost_test = sess.run(cross_entropy, {xs: x_test, ys: y_test, keep_prob: 1.0})

		loss_train.append(cost_train)
		loss_test.append(cost_test)

plt.plot(loss_train, c='r')
plt.plot(loss_test, c='b')
plt.show()
