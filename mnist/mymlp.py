from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import numpy as np

rng = np.random.RandomState(1234)
random_state = 42


def mlp(train_X, train_y, test_X, hid_dim = 100, n_epochs=10,  batch_size=100 ,eps=0.1):
    """ train by multiple layer perceptron without tensorflow
    
    arguments:
        hid_dim:     dimention of hidden layer
        n_epochs:   number of epochs
        batch_size:  batch size of stochastic gradient descent
        eps:           training rate of back propagation
    """
    
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def deriv_sigmoid(x):
        return sigmoid(x)*(1 - sigmoid(x))
    
    def get_onehot(x):
        onehot=np.zeros((len(x),10))
        onehot[np.arange(len(x)),x]=1
        return onehot

    def forward(x):
        global W1, b1, W2, b2
        
        # Forward Propagation Layer1
        u1 = np.matmul(x, W1) + b1
        z1 = sigmoid(u1)
        
        # Forward Propagation Layer2
        u2 = np.matmul(z1, W2) + b2
        z2 = softmax(u2)
        
        return u1,z1,u2,z2

    
    def train(x, t, eps):
        global W1, b1, W2, b2 # to access variables that defined outside of this function.

        #Forward Propagation
        u1,z1,u2,z2=forward(x)
        
        # Back Propagation (Cost Function: Negative Loglikelihood)
        y = z2
        cost = np.sum(get_onehot(t)*np.log(y))
        delta_2 = y - get_onehot(t) # Layer2 delta
        delta_1 = deriv_sigmoid(u1) * np.matmul(delta_2, W2.T) # Layer1 delta

        # Update Parameters Layer1
        dW1 = np.matmul(x.T, delta_1)
        db1 = np.matmul(np.ones(len(x)), delta_1)
        W1 = W1 - eps*dW1
        b1 = b1 - eps*db1

        # Update Parameters Layer2
        dW2 = np.matmul(z1.T, delta_2)
        db2 = np.matmul(np.ones(len(z1)), delta_2)
        W2 = W2 - eps*dW2
        b2 = b2 - eps*db2
        return cost

    print("BEGIN: MLP learning with hid_dim = {0}, n_epochs = {1}, batch_size = {2}, eps = {3}".format(hid_dim, n_epochs, batch_size, eps))
    
    global W1, b1, W2, b2
    # Layer1 weights
    W1 = np.random.uniform(low=-0.08, high=0.08, size=(784, hid_dim)).astype('float32')
    b1 = np.zeros(hid_dim).astype('float32')
    # Layer2 weights
    W2 = np.random.uniform(low=-0.08, high=0.08, size=(hid_dim, 10)).astype('float32')
    b2 = np.zeros(10).astype('float32')
    
    # Epoch    
    n_batches = train_X.shape[0] // batch_size
    for epoch in range(n_epochs):
        train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            cost = train(train_X[start:end], train_y[start:end], eps)

        _u1,_z1,_u2, z2=forward(train_X)
        cost = np.sum(get_onehot(train_y)*np.log(z2))
        print("\titer "+str(epoch)+" :  "+str(cost))
        
    _, _, _, pred_y = forward(test_X)
    return np.argmax(pred_y, axis=1)