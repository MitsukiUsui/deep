from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np

def load_mnist():
    """load MNIST datasets under ~/scikit_learn_data/mldata
    
    Datasets are normalized. 
    if not exist, download from sklearn repos
    """
    
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                                       mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    return train_test_split(mnist_X, mnist_y,
                test_size=0.2,
                random_state=42)

def score(model,  size=None, **kwargs ):
    """calculate F1-score　for each model
    
    arguments:
        model    :  model(train_X, train_y, test_X, **kwargs) -> pred_y
        size       :  size of training data
        **kwargs: argument for given model 
    """

    train_X, test_X, train_y, test_y = load_mnist()
    
    if size is not None:
        train_X = train_X[:size]
        train_y = train_y[:size]
        test_X = test_X[:size]
        test_y = test_y[:size]
    
    pred_y = model(train_X, train_y, test_X,  **kwargs)
    print(f1_score(test_y, pred_y, average='macro'))