from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1234)
random_state = 42


class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(fan_in + fan_out)),
            high=np.sqrt(6/(fan_in + fan_out)),
            size=filter_shape
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
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
        # Xavier Initialization
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(in_dim + out_dim)),
            high=np.sqrt(6/(in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

    

def cnn(train_X, train_y, test_X, n_epochs =50, batch_size = 100, eps = 0.01):
    """ train by convolutional neural network with tensorflow
    
    arguments:
        n_epochs:   number of epochs
        batch_size:  batch size of stochastic gradient descent
        eps:           training rate of back propagation
    """
    
    def get_onehot(x):
        onehot=np.zeros((len(x),10))
        onehot[np.arange(len(x)),x]=1
        return onehot
    
    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x
    
    layers = [                            # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((5, 5, 1, 20), tf.nn.relu),  # 28x28x 1 -> 24x24x20
        Pooling((1, 2, 2, 1)),            # 24x24x20 -> 12x12x20
        Conv((5, 5, 20, 50), tf.nn.relu), # 12x12x20 ->  8x 8x50
        Pooling((1, 2, 2, 1)),            #  8x 8x50 ->  4x 4x50
        Flatten(),
        Dense(4*4*50, 10, tf.nn.softmax)
    ]

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    t = tf.placeholder(tf.float32, [None, 10])

    y = f_props(layers, x)
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
    train = tf.train.GradientDescentOptimizer(eps).minimize(cost)
    valid = tf.argmax(y, 1)
    

    print("BEGIN: CNN learning with n_epochs = {0}, batch_size = {1}, eps = {2}".format(n_epochs, batch_size, eps))
    
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
    train_y=get_onehot(train_y)
    
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)
    n_batches = train_X.shape[0]//batch_size

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
            print('\tEPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
        
        pred_y= sess.run(valid, feed_dict={x: test_X})
    return pred_y