from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1234)
random_state = 42


def mlp(train_X, train_y, test_X, hid_dim = 100, n_epochs=10,  batch_size=100 ,eps=0.1):
    """ train by multiple layer perceptron with tensorflow
    
    arguments:
        hid_dim:     dimention of hidden layer
        n_epochs:   number of epochs
        batch_size:  batch size of stochastic gradient descent
        eps:           training rate of back propagation
    """

    def get_onehot(x):
        onehot=np.zeros((len(x),10))
        onehot[np.arange(len(x)),x]=1
        return onehot
    
    
    # Step1. プレースホルダーと変数の定義
    ## Placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    t = tf.placeholder(tf.float32, [None, 10])

    ## 変数
    W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(784, hid_dim)).astype('float32'), name='W1')
    b1 = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b1')
    W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(hid_dim, 10)).astype('float32'), name='W2')
    b2 = tf.Variable(np.zeros(10).astype('float32'), name='b2')
    params = [W1, b1, W2, b2]

    
    # Step2. グラフの定義
    u1 = tf.matmul(x, W1) + b1
    z1 = tf.nn.sigmoid(u1)
    u2 = tf.matmul(z1, W2) + b2
    y = tf.nn.softmax(u2)
    valid = tf.argmax(y, 1)

    # Step3. 誤差関数の定義
    cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))) # tf.log(0)によるnanを防ぐ

    # Step4. 更新則の設定
    gW1, gb1, gW2, gb2 = tf.gradients(cost, params)
    updates = [
        W1.assign_add(-eps*gW1),
        b1.assign_add(-eps*gb1),
        W2.assign_add(-eps*gW2),
        b2.assign_add(-eps*gb2)
    ]
    train = tf.group(*updates)

    # Step5. 学習
    print("BEGIN: MLP learning with hid_dim = {0}, n_epochs = {1}, batch_size = {2}, eps = {3}".format(hid_dim, n_epochs, batch_size, eps))
    
    train_y=get_onehot(train_y)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=random_state)

    n_batches = train_X.shape[0] // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
            print('\tEPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
            pred_y = sess.run(valid, feed_dict={x: test_X}) 
    return pred_y