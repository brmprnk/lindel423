import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mca import MCA
import sys
import pickle as pkl
import os, sys, csv, re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np
from datetime import datetime

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import *
# Make the data preprocessing a deterministic process
np.random.seed(42)

np.set_printoptions(threshold=sys.maxsize)

## If data already preprocessed, load it in, else do preprocessing

try:
    data = pd.read_csv('./data/Lindel_alldata.csv', index_col=0)
    print("Data successfully loaded")
except Exception:
    label, rev_index, features = pkl.load(open('./data/feature_index_all.pkl','rb'))
    Lindel_training = pd.read_csv("./data/Lindel_training_65bp.csv", sep=',', index_col=0)
    Lindel_test = pd.read_csv("./data/Lindel_test_65bp.csv", sep=',', index_col=0)

    print("Number of labels : ", len(label.keys()))
    print("Number of rev_index : ", len(rev_index.keys()))
    print("Number of features : ", len(features.keys()))

    # column descriptions
    # Lindel_training.iloc[0] # guide sequences
    # Lindel_training.iloc[1:3034] # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
    # Lindel_training.iloc[3034:] # 557 observed outcome frequencies

    # # Merge training and test set for dimensionality reduction
    all_data = pd.concat([Lindel_training, Lindel_test])
    # data_features = all_data.iloc[:, 1:3034]

    # # Clean up data
    features = dict(sorted(features.items(), key=lambda item: item[1]))
    feature_labels = list(features.keys())

    labels = dict(sorted(label.items(), key=lambda item: item[1]))
    class_labels = list(labels.keys())

    one_hot_labels = []
    for i in range(80):
        one_hot_labels.append("nt {}".format(str(int(i / 4) + 1)))

    for i in range(304):
        one_hot_labels.append("2nt {}".format(str(int(i / 16) + 1)))

    one_hot_labels = np.array(one_hot_labels)

    column_labels = np.concatenate((np.array(['Guide Sequence', '65bp']), feature_labels, one_hot_labels, class_labels))

    # Rename columns of test and training set
    Lindel_training = Lindel_training.set_axis(column_labels, axis=1, inplace=False)
    Lindel_test = Lindel_test.set_axis(column_labels, axis=1, inplace=False)

    data = pd.concat([Lindel_training, Lindel_test], axis=0)

# Do data preprocessing
# Define useful functions
def mse(x, y):
    return ((x - y) ** 2).mean()


def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


def onehotencoder(seq):
    nt = ['A', 'T', 'C', 'G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i] + str(k))

    for k in range(l - 1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i] + nt[j] + str(k))
    head_idx = {}
    for idx, key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j] + str(j)]] = 1.
    for k in range(l - 1):
        encode[head_idx[seq[k:k + 2] + str(k)]] = 1.
    return encode


def kfoldsplits(X):
    """Split annotations"""
    kf = KFold(n_splits=10, shuffle=False)
    splits = []
    for trainIdx, validIdx in kf.split(X):
        splits.append((trainIdx, validIdx))

    print("The first index of the first split is ", splits[0][0][0])

    return splits


# Preprocess data
model_data = data.values[:, 2:].astype(np.float32)
print(model_data.shape, type(model_data))

# Sum up deletions and insertions to
X = model_data[:, :(2649 + 384)]
y = model_data[:, (2649 + 384):]

print("X Shape ", X.shape, " | y shape ", y.shape)

# Randomly shuffle data
idx = np.arange(len(y))
np.random.shuffle(idx)
X, y = X[idx], y[idx]

print("Now removing samples with only insertion events")
X_deletion, y_deletion = [], []

# Remove samples that only have insertion events:
for i in range(model_data.shape[0]):
    if 1 > sum(y[i, :536]) > 0:
        y_deletion.append(y[i, :536] / sum(y[i, :536]))
        X_deletion.append(X[i])

X_deletion, y_deletion = np.array(X_deletion), np.array(y_deletion)

print("X_deletion Shape ", X_deletion.shape, " | y_deletion shape ", y_deletion.shape)

splits = kfoldsplits(X_deletion)
print("Number of train/val splits: ", len(splits))

# Make results dir
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
save_dir = os.path.join('./results', dt_string)
os.makedirs(save_dir)

print("Save dir is ", save_dir)

# Train model: No regularization, no early stopping
for i in tqdm(range(len(splits))):
    print("Baseline ", (i + 1), "of 10")

    train_split, val_split = splits[i]

    x_train = X_deletion[train_split]
    x_valid = X_deletion[val_split]

    y_train = y_deletion[train_split]
    y_valid = y_deletion[val_split]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="{}/logs/baseline".format(save_dir))

    checkpoint_name = save_dir + '/baseline_cp{}'.format(i)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_name + '-{epoch:02d}.h5',
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="min")

    csv_logger = CSVLogger('{}/baseline{}.log'.format(save_dir, i), separator=',', append=False)

    model = Sequential()
    model.add(Dense(536, activation='softmax', input_shape=(X_deletion.shape[1],), kernel_regularizer=None))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[
                            tensorboard_callback,
                            model_checkpoint_callback,
                            csv_logger], verbose=0)

# Train model: L1 and L2, no early stopping
baseline_errors = []
for i in range(len(splits)):
    print("L1 / L2 ", (i + 1), "of 10")

    train_split, val_split = splits[i]

    x_train = X_deletion[train_split]
    x_valid = X_deletion[val_split]

    y_train = y_deletion[train_split]
    y_valid = y_deletion[val_split]

    # L1 Regularization
    lambdas = 10 ** np.arange(-10, -1, 0.1)

    for l in lambdas:
        tensorboard_callback_l1 = tf.keras.callbacks.TensorBoard(log_dir="{}/logs/l1".format(save_dir))

        checkpoint_name = save_dir + '/l1_{}_lambda_{}'.format(i, l)

        csv_logger = CSVLogger('{}/L1_{}.log'.format(save_dir, i), separator=',', append=False)

        model = Sequential()
        model.add(Dense(536, activation='softmax', input_shape=(X_deletion.shape[1],), kernel_regularizer=l1(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), verbose=0,
                            callbacks=[tensorboard_callback_l1, csv_logger,
                                       tf.keras.callbacks.ModelCheckpoint(
                                           filepath=checkpoint_name + '-{epoch:02d}.h5',
                                           monitor="val_loss",
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode="min"
                                       )
                                       ])

    # L2 Regularization
    for l in lambdas:
        tensorboard_callback_l2 = tf.keras.callbacks.TensorBoard(log_dir="{}/logs/l2".format(save_dir))

        checkpoint_name = save_dir + '/l2_{}_lambda_{}'.format(i, l)

        csv_logger = CSVLogger('{}/L2_{}.log'.format(save_dir, i), separator=',', append=False)

        model = Sequential()
        model.add(Dense(536, activation='softmax', input_shape=(X_deletion.shape[1],), kernel_regularizer=l2(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), verbose=0,
                            callbacks=[tensorboard_callback_l2, csv_logger,
                                       tf.keras.callbacks.ModelCheckpoint(
                                           filepath=checkpoint_name + '-{epoch:02d}.h5',
                                           monitor="val_loss",
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode="min"
                                       )
                                       ])
