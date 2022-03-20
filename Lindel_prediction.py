#!/usr/bin/env python 
# Author: Will Chen
''' 
1. All functions are tested under python3.5 and python 3.6
2. Add Lindel folder to your python path.
3. y_hat is the prediction of all ~450 classes of indels <30bp.
4. fs is the frameshift ratio for this sequence.
5. Input should be 65bp (30 bp upstream and 35 bp downstream of the cleavage site)
usage: pyton Lindel_predction.py your_sequence_here your_file_name_here(can be gene name or guide name you designed)
'''
import Lindel, os, sys
from Lindel.Predictor import * 
import pickle as pkl
import pandas as pd

model_weights = pkl.load(open(os.path.join("./Lindel/Given_Model_Weights.pkl"),'rb'))

# label: dict where keys are the outcome classes and the values are the corresponding indices of columns
# rev_index: dict where the keys are the indices of columns and the values are the corresponding outcome classes
# features: dict where the keys are the features and the values are the indices
label, rev_index, features = pkl.load(open(os.path.join('./data/feature_index_all.pkl'),'rb'))

# Load in Data Matrix
data = pd.read_csv("./data/Lindel_test_65bp.csv", sep=',', index_col=0)

predictions = {}

# Generate predictions for each sequence
for index, entry in data.iterrows():

    # Guide sequence
    guide = str(entry[0])
    seq = str(entry[1])
    features = entry[2:3035].values.astype(np.float32)
    labels = entry[3035:].values.astype(np.float32)

    y_hat = gen_prediction(guide, seq, features, label, model_weights)

    # Get predictions for each of the 557 classes (probabilities sum up to 1)
    pred_freq = {}
    for i in range(len(y_hat)):
        if y_hat[i]!=0:
            pred_freq[rev_index[i]] = y_hat[i]

    # Predictions in order of rev_index
    predictions[guide] = np.array(list(pred_freq.values()))
    
    # Sort predictions in descending order
    pred_sorted = sorted(pred_freq.items(), key=lambda kv: kv[1],reverse=True)

    # TODO: Do something with your predictions!

# Save predictions to pickle
with open('predictions_testset_givenweights_withcmax.pkl', 'wb') as handle:
    pkl.dump(predictions, handle, protocol=pkl.HIGHEST_PROTOCOL)