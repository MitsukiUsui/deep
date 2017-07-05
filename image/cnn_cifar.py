from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.sequence import pad_sequences

import sys
import os
import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1234)
random_state=42
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def gcn(x):
    mean = np.mean(x, axis=(1, 2, 3), keepdims=True)#データ数の軸で標準化を行う
    std = np.std(x, axis=(1, 2, 3), keepdims=True)
    return (x - mean)/std#分散が0になる場合の処理は今回はなし

class ZCAWhitening:
    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.mean = None
        self.ZCA_matrix = None

    def fit(self, x):
        x = x.reshape(x.shape[0], -1)
        self.mean = np.mean(x, axis=0)
        x -= self.mean
        cov_matrix = np.dot(x.T, x) / x.shape[0]
        A, d, _ = np.linalg.svd(cov_matrix)
        self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.epsilon))), A.T)

    def transform(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x -= self.mean
        x = np.dot(x, self.ZCA_matrix.T)
        return x.reshape(shape)


class BatchNorm:
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.gamma = tf.Variable(np.ones(shape, dtype='float32'), name='gamma')
        self.beta  = tf.Variable(np.zeros(shape, dtype='float32'), name='beta')
        self.epsilon = epsilon#０でわられることを防ぐためのパラメーター

    def f_prop(self, x):
        if len(x.get_shape()) == 2:
            mean, var = tf.nn.moments(x, axes=0, keepdims=True)
            std = tf.sqrt(var + self.epsilon)
        elif len(x.get_shape()) == 4:
            mean, var = tf.nn.moments(x, axes=(0,1,2), keep_dims=True)#momentsを使うと平均と分散を計算してくれる。
            std = tf.sqrt(var + self.epsilon)
            normalized_x = (x - mean) / std
            return self.gamma * normalized_x + self.beta    

        
class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
        # Xavier
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(#一様分布から初期化
            low=-np.sqrt(6/(fan_in + fan_out)),
            high=np.sqrt(6/(fan_in + fan_out)),
            size=filter_shape
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごと
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)

class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)


class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(in_dim + out_dim)),
            high=np.sqrt(6/(in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

class Activation:
    def __init__(self, function=lambda x: x):
        self.function = function

    def f_prop(self, x):
        return self.function(x)        

        
        
    
def cnn(train_X, train_y, test_X, n_epochs = 20, batch_size = 100):
  
    #--------------------------------------------------------------------------------
	# graph construction
	#--------------------------------------------------------------------------------
    
    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x
    
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    t = tf.placeholder(tf.float32, [None, 10])
    
    layers = [ # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((3, 3, 3, 32)), # 32x32x3 -> 30x30x32
        BatchNorm((30, 30, 32)),
        Activation(tf.nn.relu),
        Pooling((1, 2, 2, 1)), # 30x30x32 -> 15x15x32
        Conv((3, 3, 32, 64)), # 15x15x32 -> 13x13x64
        BatchNorm((13, 13, 64)),
        Pooling(((1, 2, 2, 1))), # 13x13x64 -> 6x6x64
        Conv((3, 3, 64, 128)), # 6x6x64 -> 4x4x128
        BatchNorm((4, 4, 128)),
        Activation(tf.nn.relu),
        Pooling((1, 2, 2, 1)), # 4x4x128 -> 2x2x128
        Flatten(),
        Dense(2*2*128, 256, tf.nn.relu),
        Dense(256, 10, tf.nn.softmax)
    ]
    
    
    y = f_props(layers, x)
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    valid = tf.argmax(y, 1)
    
    
    
    #--------------------------------------------------------------------------------
	# preprocess
	#--------------------------------------------------------------------------------
    
    print("BEGIN: cnn with  n_epochs = {0}, batch_size = {1}".format(n_epochs, batch_size))
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=random_state)
    
    ##preprocess
    zca = ZCAWhitening()
    zca.fit(gcn(train_X))
    zca_train_X = zca.transform(gcn(train_X))
    zca_train_y = train_y[:]
    zca_valid_X = zca.transform(gcn(valid_X))
    zca_valid_y = valid_y[:]
    zca_test_X = zca.transform(gcn(test_X))
    print("DONE: preprocess")
    

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    n_batches = train_X.shape[0]//batch_size
    
    for epoch in range(n_epochs):
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: zca_train_X[start:end], t: zca_train_y[start:end]})
        pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: zca_valid_X, t: zca_valid_y})
        print('\tEPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))

    pred_y = sess.run(valid, feed_dict={x: zca_test_X})
    return pred_y

    

def main():
  
    (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]
    cifar_X = cifar_X.astype('float32') / 255
    cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y,
                                                        test_size=10000,
                                                        random_state=random_state)
    
    pred_y = cnn(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))

    
if __name__=="__main__":
	main() 
