from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

import sys
import os
import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1234)
random_state=42
padding_value=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#--------------------------------------------------------------------------------
#  Random orthogonal initializer (see [Saxe et al. 2013])
#--------------------------------------------------------------------------------
def orthogonal_initializer(shape, scale = 1.0):
    a = np.random.normal(0.0, 1.0, shape).astype(np.float32)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    return scale * q


class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(rng.randn(vocab_size, emb_dim).astype('float32') * scale, name='V')

    def f_prop(self, x):
        return tf.nn.embedding_lookup(self.V, x)
    
class RNN:
    def __init__(self, in_dim, hid_dim, m, scale=0.08):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        # Xavier initializer
        self.W_in = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(in_dim + hid_dim)),
            high=np.sqrt(6/(in_dim + hid_dim)),
            size=(in_dim, hid_dim)
        ).astype('float32'), name='W_in')
        # Random orthogonal initializer
        self.W_re = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_re')
        self.b_re = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_re')
        self.m = m

    def f_prop(self, x):
        def fn(h_tm1, x_and_m):
            x = x_and_m[0]
            m = x_and_m[1]
            h_t = tf.nn.tanh(tf.matmul(h_tm1, self.W_re) + tf.matmul(x, self.W_in) + self.b_re)
            return m[:, None] * h_t + (1 - m[:, None]) * h_tm1 # Mask

        # shape: [batch_size, sentence_length, in_dim] -> shape: [sentence_length, batch_size, in_dim]
        _x = tf.transpose(x, perm=[1, 0, 2])
        # shape: [batch_size, sentence_length] -> shape: [sentence_length, batch_size]
        _m = tf.transpose(self.m)
        h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state

        h = tf.scan(fn=fn, elems=[_x, _m], initializer=h_0)

        return h[-1] # Take the last state

    
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier initializer
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(in_dim + out_dim)),
            high=np.sqrt(6/(in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)    
    
    
def rnn(train_X, train_y, test_X,  num_words, emb_dim=100, hid_dim=50, n_epochs=5, batch_size=100):
    def f_props(layers, x):
        for i, layer in enumerate(layers):
            x = layer.f_prop(x)
        return x

	#--------------------------------------------------------------------------------
	# layer configuration
	#--------------------------------------------------------------------------------
    x = tf.placeholder(tf.int32, [None, None], name='x')
    m = tf.cast(tf.not_equal(x, padding_value), tf.float32) # Mask. Paddingの部分(padding_value)は0, 他の値は1
    t = tf.placeholder(tf.float32, [None, None], name='t')

    layers = [
        Embedding(num_words, emb_dim),
        RNN(emb_dim, hid_dim, m=m),
        Dense(hid_dim, 1, tf.nn.sigmoid)
    ]
    y = f_props(layers, x)
    cost = tf.reduce_mean(-t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)) - (1. - t)*tf.log(tf.clip_by_value(1.-y, 1e-10, 1.0)))
    train = tf.train.AdamOptimizer().minimize(cost)
    test = tf.round(y)

 
   	#--------------------------------------------------------------------------------
	#learning main
	#--------------------------------------------------------------------------------
    # split data into training and validation
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=random_state)

    # Sort train data according to its length
    train_X_lens = [len(com) for com in train_X]
    sorted_train_indexes = sorted(range(len(train_X_lens)), key=lambda x: -train_X_lens[x])
    train_X = [train_X[idx] for idx in sorted_train_indexes]
    train_y = [train_y[idx] for idx in sorted_train_indexes]
    
    n_batches_train = len(train_X) // batch_size
    n_batches_valid = len(valid_X) // batch_size
    n_batches_test  = len(test_X) // batch_size

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(n_epochs):
            # Train
            train_costs = []
            for i in range(n_batches_train):
                start = i * batch_size
                end = start + batch_size

                train_X_mb = np.array(pad_sequences(train_X[start:end], padding='post', value=padding_value)) # Padding
                train_y_mb = np.array(train_y[start:end])[:, np.newaxis]

                _, train_cost = sess.run([train, cost], feed_dict={x: train_X_mb, t: train_y_mb})
                train_costs.append(train_cost)

            # Valid
            valid_costs = []
            pred_y = []
            for i in range(n_batches_valid):
                start = i * batch_size
                end = start + batch_size

                valid_X_mb = np.array(pad_sequences(valid_X[start:end], padding='post', value=padding_value)) # Padding
                valid_y_mb = np.array(valid_y[start:end])[:, np.newaxis]

                pred, valid_cost = sess.run([test, cost], feed_dict={x: valid_X_mb, t: valid_y_mb})
                pred_y += pred.flatten().tolist()
                valid_costs.append(valid_cost)
            print('EPOCH: %i, Training cost: %.3f, Validation cost: %.3f, Validation F1: %.3f' % (epoch+1, np.mean(train_costs), np.mean(valid_costs), f1_score(valid_y, pred_y, average='macro')))
    
        # Test
        pred_y=[]
        for i in range(n_batches_test):
            start = i * batch_size
            end = start + batch_size
            test_X_mb = np.array(pad_sequences(test_X[start:end], padding='post', value=padding_value)) # Padding
            pred = sess.run(test, feed_dict={x: test_X_mb})
            pred_y+=pred.flatten().tolist()
    return pred_y


def main():
    num_words=10000
    emb_dim=100
    hid_dim=50
    n_epochs=5
    batch_size=100
    
    # load IMDb data sets(0: for padding, 1:start_char, 2:oov_char, 3~:word_index)
    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=num_words, seed=42, start_char=1, oov_char=2, index_from=3)
    
    pred_y = rnn(train_X, train_y, test_X,  num_words=num_words)
    true_y =  test_y.tolist()
    print(f1_score(true_y, pred_y, average='macro'))
    
if __name__=="__main__":
	main() 
