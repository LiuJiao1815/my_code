import numpy as np 
np.random.seed(1337)   #for reproducibility

from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Dense, Input 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


#X shape(60000, 28 * 28) , y shape(10000, )
(x_train, _), (x_test, y_test ) = mnist.load_data()

#数据预处理
x_train = x_train.astype('float32')/255. - 0.5    #minmax_normalized
x_test = x_test.astype('float32')/255. - 0.5      #minmax_normalized

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)
print(x_test.shape)

#压缩特征维度到两维
#encoding_dim = 2

encoding_dim = 3
#this is our input placeholder
input_img = Input(shape = (784,))

#编码层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation = 'relu')(encoded)
encoded = Dense(10, activation = 'relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

#解码层
decoded = Dense(10, activation = 'relu')(encoder_output)
decoded = Dense(64, activation = 'relu')(decoded)
decoded = Dense(128, activation = 'relu')(decoded)
decoded = Dense(784, activation = 'tanh')(decoded)

#构建自编码模型
autoencoder = Model(input_img, outputs = decoded)

#构建编码模型
encoder = Model(inputs= input_img, outputs = encoder_output)

#complie autoencoder
autoencoder.compile(optimizer = 'adam', loss = 'mse')

#training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle = True)



# plotting
encoded_imgs = encoder.predict(x_test)

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2],c=y_test) 
plt.show()

#ploting

'''
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0], encoded_imgs[:,1], c=y_test, s=3)
plt.colorbar()
plt.show()
'''
