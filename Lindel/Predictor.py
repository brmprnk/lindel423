import numpy as np
import scipy.sparse as sparse

def gen_indel(sequence,cut_site):
    '''This is the function that used to generate all possible unique indels and
    list the redundant classes which will be combined after'''
    nt = ['A','T','C','G']
    up = sequence[0:cut_site]
    down = sequence[cut_site:]
    dmax = min(len(up),len(down))
    uniqe_seq ={}
    for dstart in range(1,cut_site+3):
        for dlen in range(1,dmax):
            if len(sequence) > dlen+dstart > cut_site-2:
                seq = sequence[0:dstart]+sequence[dstart+dlen:]
                indel = sequence[0:dstart] + '-'*dlen + sequence[dstart+dlen:]
                array = [indel,sequence,13,'del',dstart-30,dlen,None,None,None]
                try:
                    uniqe_seq[seq]
                    if dstart-30 <1:
                        uniqe_seq[seq] = array
                except KeyError: uniqe_seq[seq] = array
    for base in nt:
        seq = sequence[0:cut_site]+base+sequence[cut_site:]
        indel = sequence[0:cut_site]+'-'+sequence[cut_site:]
        array = [sequence,indel,13,'ins',0,1,base,None,None]
        try: uniqe_seq[seq] = array
        except KeyError: uniqe_seq[seq] = array
        for base2 in nt:
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site]+'--'+sequence[cut_site:]
            array = [sequence,indel,13,'ins',0,2,base+base2,None,None]
            try: uniqe_seq[seq] = array
            except KeyError:uniqe_seq[seq] = array
    uniq_align = label_mh(list(uniqe_seq.values()),4)
    for read in uniq_align:
        if read[-2]=='mh':
            merged=[]
            for i in range(0,read[-1]+1):
                merged.append((read[4]-i,read[5]))
            read[-3] = merged
    return uniq_align

def label_mh(sample,mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        if read[3] == 'del':
            idx = read[2] + read[4] +17
            idx2 = idx + read[5]
            x = mh_len if read[5] > mh_len else read[5]
            for i in range(x,0,-1):
                if read[1][idx-i:idx] == read[1][idx2-i:idx2] and i <= read[5]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break
            if sample[k][-2]!='mh':
                sample[k][-1]=0
    return sample


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


def gen_prediction(guide, seq, features, label, model_weights):
    '''generate the prediction for all classes, redundant classes will be combined'''

    w1,b1,w2,b2,w3,b3 = model_weights

    cmax = gen_cmatrix(gen_indel(seq, 30), label)  # combine redundant classes

    # Get one-hot-encoded features
    input_indel = features[-384:]

    # Get 104 binary features
    # FROM THE PAPER, but what are these?:
    # "We split 2680 targets associated with both insertion and deletion outcomes
    # into training (n = 2000) and test (n = 680) sets, and trained a linear regression model
    # to predict the proportion of insertion events based on position-specific content of the
    # hexamer centered on the DSB (single and dinucleotide k-mers; 104 binary features; Figure 3E).
    # The model performs reasonably well (Pearson's r = 0.70).
    input_ins   = onehotencoder(guide[-6:])

    # Get ratio of deletions and insertions
    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)

    # Deletions
    deletions  = softmax(np.dot(features, w2) + b2)

    # Insertions
    insertions = softmax(np.dot(input_ins, w3) + b3)

    # Scale deletions and insertions by their ratio's

    y_hat = np.concatenate((deletions * dratio, insertions * insratio), axis=None).astype(np.float32) * cmax

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