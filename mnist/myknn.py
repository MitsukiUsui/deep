from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np


def knn(train_X, train_y, test_X, kMax=20):
    """ train by k-nearest neighbor
    
    arguments:
        kMax: k will be searched among [1,kMax]
    """
    
    def calculate_index_matrix(dist_mat, kMax):
        """given distance matrix, return top kMax index for each validate date
        
        arguments:
            dist_mat : [validateSize, trainSize]
        return:
            idx_mat : [validateSize, kMax]
        """
        
        validateSize=dist_mat.shape[0]
        idx_mat=-np.ones((validateSize, kMax)).astype(int)
        for i in range(validateSize):
            topK=np.argpartition(dist_mat[i,:], -kMax)[-kMax:]#choose top kMax indexs
            idx_mat[i,:]=topK[np.argsort(dist_mat[i,topK])]#sort exactly with top kMax
        return idx_mat
    
    print("BEGIN: k-NN learning with k-max = {}".format(kMax))
    
    #normalize
    norm = np.linalg.norm(train_X, ord=2, axis=1)
    n_train_X = train_X / norm[:, np.newaxis]
    
    n_train_X, n_valid_X, train_y, valid_y=train_test_split(n_train_X, train_y, test_size=0.2)
    
    #calculate idx_mat(valid, kMax)
    dist_mat=np.dot(n_valid_X, n_train_X.T)
    idx_mat=calculate_index_matrix(dist_mat, kMax)
   
    #calculate f1score for each k
    score_arr=-np.ones(kMax+1)#score_arr[0] is dammy
    for k in range(1, kMax+1):
        validateSize=n_valid_X.shape[0]
        pred_y=np.zeros(validateSize).astype(int)
        for i in range(len(pred_y)):
            idx=idx_mat[i,-k:]
            counts=np.bincount(train_y[idx])
            pred_y[i]=np.argmax(counts)
        score_arr[k]=f1_score(valid_y, pred_y, average='macro')
        print("\tk = {} : f1-score = {}".format(k, score_arr[k]))
        
    #choose best k
    k=np.argmax(score_arr)
    #print("k = {} is choosen as best".format(k))
    
    #calculate  pred_y with defined k
    norm = np.linalg.norm(test_X, ord=2, axis=1)
    n_test_X = test_X / norm[:, np.newaxis]
    dist_mat=np.dot(n_test_X, n_train_X.T)
    idx_mat=calculate_index_matrix(dist_mat, k)
    pred_y=np.zeros(n_test_X.shape[0]).astype(int)
    for i in range(len(pred_y)):
        counts=np.bincount(train_y[idx_mat[i,:]])
        pred_y[i]=np.argmax(counts)   
        
    print("DONE: prediction with k = {}".format(k))
    return pred_y