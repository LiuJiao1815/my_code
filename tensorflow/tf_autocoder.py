import tensorflow as tf 
from tensorflow.examples.tutorials.mnist input_data
mnist = input_data.read_data_stes('F:\\Python\\python_tensorflow', one_hot = False)

#parameters
learning_rate = 0.001
train_rate = 20
batch_size = 256
display_step = 1

#network parameters
num_input = 784

#placeholder
x = tf.placeholder(tf.float32, [None, num_input])

#hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
	'encoder_hidden_1' : tf.Variables(tf.random_normal([num_input, n_hidden_1])),
	'encoder_hidden_2' : tf.Variables(tf.random_normal([n_hidden_1, n_hidden_2])),
	'encoder_hidden_3' : tf.Variables(tf.random_normal([n_hidden_2, n_hidden_3])),
	'encoder_hidden_4' : tf.Variables(tf.random_normal([n_hidden_3, n_hidden_4])),

	'decoder_hidden_1' : tf.Variables(tf.random_normal([n_hidden_4, n_hidden_3])),
	'decoder_hidden_2' : tf.Variables(tf.random_normal([n_hidden_3, n_hidden_2])),
	'decoder_hidden_3' : tf.Variables(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_hidden_4' : tf.Variables(tf.random_normal([n_hidden_1, num_input ])) 
}

biases = {
	'encoder_hidden_1' : tf.Variables(tf.random_normal([n_hidden_1]))
	'encoder_hidden_2' : tf.Variables(tf.random_normal([n_hidden_2]))
	'encoder_hidden_3' : tf.Variables(tf.random_normal([n_hidden_3]))
	'encoder_hidden_4' : tf.Variables(tf.random_normal([n_hidden_4]))

	'decoder_hidden_1' : tf.Variables(tf.random_normal([n_hidden_1]))
	'decoder_hidden_2' : tf.Variables(tf.random_normal([n_hidden_2]))
	'decoder_hidden_3' : tf.Variables(tf.random_normal([n_hidden_3]))
	'decoder_hidden_4' : tf.Variables(tf.random_normal([n_hidden_4]))

}

#bulid the decoder

def encoder(x) :
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_hidden_1']), biases['encoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_hidden_2']), biases['encoder_hidden_2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_hidden_3']), biases['encoder_hidden_3']))
	layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_hidden_4']), biases['encoder_hidden_4'])
	return layer_4

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_hidden_1']), biases['decoder_hidden_1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_hidden_2']), biases['decoder_hidden_2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_hidden_3']), biases['decoder_hidden_3']))
	layer_4 = tf.nn