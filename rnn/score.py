from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

import sys
import numpy as np
import tensorflow as tf

    
def score(model,  num_words=10000, size=None,  **kwargs ):
    """calculate F1-score　for each model on IMDb's sentiment analysis
    
    arguments:
        model        :  model(train_X, train_y, test_X, **kwargs) -> pred_y
        num_words : 
        size           :  size of training data
        **kwargs    : arguments for given model 
    """
    
    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=num_words, seed=42, start_char=0, oov_char=1, index_from=2)
    
    if size is not None:
        train_X = train_X[:size]
        train_y = train_y[:size]
        test_X = test_X[:size]
        test_y = test_y[:size]
    
    pred_y = model(train_X, train_y, test_X,  num_words=num_words, **kwargs)
    true_y =  test_y.tolist()
    print(f1_score(true_y, pred_y, average='macro'))


