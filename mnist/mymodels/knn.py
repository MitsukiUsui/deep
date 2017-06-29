from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np


def knn(train_X, train_y, test_X, k_max=20):
    """ train by k-nearest neighbor
    
    arguments:
        k_max: k will be searched among [1,k_max]
    """
    
    def calculate_index_matrix(n_train_X, n_test_X, k_max):
        """given distance matrix, return top k_max index for each validate date
        
        arguments:
            n_train_X : [train_size, 784]
            n_test_X  : [test_size, 784]
        return:
            idx_mat : [test_size, k_max]
        """
        
        dist_mat=np.dot(n_test_X, n_train_X.T)
        test_size=n_test_X.shape[0]
        idx_mat=-np.ones((test_size, k_max)).astype(int)
        for i in range(test_size):
            top_k=np.argpartition(dist_mat[i,:], -k_max)[-k_max:]#choose top k_max indexs
            idx_mat[i,:]=top_k[np.argsort(dist_mat[i,top_k])]#sort exactly with top k_max
        return idx_mat
    
    print("BEGIN: k-NN learning with k-max = {}".format(k_max))
    
    #preprocess (normalize, split)
    norm = np.linalg.norm(train_X, ord=2, axis=1)
    n_train_X = train_X / norm[:, np.newaxis]
    n_train_X, n_valid_X, train_y, valid_y=train_test_split(n_train_X, train_y, test_size=0.2)
    
    #choose top k_max nearest neighbor for each dataset
    idx_mat=calculate_index_matrix(n_train_X, n_valid_X, k_max) #[validate_size, k_max]
   
    #calculate f1score for each k in [1, k_max]
    score_arr=-np.ones(k_max+1)#score_arr[0] is dammy
    for k in range(1, k_max+1):
        validata_size=n_valid_X.shape[0]
        pred_y=np.zeros(validata_size).astype(int)
        for i in range(len(pred_y)):
            idx=idx_mat[i,-k:]
            counts=np.bincount(train_y[idx])
            pred_y[i]=np.argmax(counts)
        score_arr[k]=f1_score(valid_y, pred_y, average='macro')
        print("\tk = {} : f1-score = {}".format(k, score_arr[k]))
        
    #choose best k
    k=np.argmax(score_arr)
    
    #calculate  pred_y with defined k
    norm = np.linalg.norm(test_X, ord=2, axis=1)
    n_test_X = test_X / norm[:, np.newaxis]
    idx_mat=calculate_index_matrix(n_train_X, n_test_X, k)
    pred_y=np.zeros(n_test_X.shape[0]).astype(int)
    for i in range(len(pred_y)):
        counts=np.bincount(train_y[idx_mat[i,:]])
        pred_y[i]=np.argmax(counts)   
        
    print("DONE: prediction with k = {}".format(k))
    return pred_y
