import numpy as np
import scipy.sparse as sparse
import re
import json


def onehotencoder(seq):
    '''convert to single and di-nucleotide hotencode
    REQUIRED FOR gen_prediction
    '''
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode


def gen_prediction(seq, features, model_weights):
    '''generate the prediction for all classes, redundant classes will be combined'''

    w1,b1,w2,b2,w3,b3 = model_weights

    # Get one-hot-encoded features
    input_indel = features[-384:]

    # Get 104 binary features
    # FROM THE PAPER, but what are these?:
    # "We split 2680 targets associated with both insertion and deletion outcomes
    # into training (n = 2000) and test (n = 680) sets, and trained a linear regression model
    # to predict the proportion of insertion events based on position-specific content of the
    # hexamer centered on the DSB (single and dinucleotide k-mers; 104 binary features; Figure 3E).
    # The model performs reasonably well (Pearson's r = 0.70).
    input_ins   = onehotencoder(seq[-6:])

    # Get ratio of deletions and insertions
    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)

    # Deletions
    deletions  = softmax(np.dot(features, w2) + b2)

    # Insertions
    insertions = softmax(np.dot(input_ins, w3) + b3)

    # Scale deletions and insertions by their ratio's

    y_hat = np.concatenate((deletions * dratio, insertions * insratio), axis=None).astype(np.float64)

    return y_hat

def softmax(weights):
    return (np.exp(weights)/sum(np.exp(weights)))

def gen_cmatrix(indels,label): 
    ''' Combine redundant classes based on microhomology, matrix operation
    TODO: Should we use this??
    '''
    combine = []
    for s in indels:
        if s[-2] == 'mh':
            tmp = []
            for k in s[-3]:
                try:
                    tmp.append(label['+'.join(list(map(str,k)))])
                except KeyError:
                    pass
            if len(tmp)>1:
                combine.append(tmp)
    temp = np.diag(np.ones(557), 0)
    for key in combine:
        for i in key[1:]:
            temp[i,key[0]] = 1
            temp[i,i]=0    
    return (sparse.csr_matrix(temp))