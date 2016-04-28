#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python fofe_feature_cnn.py


The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
The hydrophilicity values are from PNAS, 1981, 78:3824-3828
(T.P.Hopp & K.R.Woods). The side-chain mass for each of the 20 amino acids. CRC
Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton,
Florida (1985). R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones,
Data for Biochemical Research 3rd ed.,
Clarendon Press Oxford (1986).

"""

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.noise import GaussianNoise, GaussianDropout

from keras.layers.core import Dense, Dropout, Activation, Highway, MaxoutDense, ActivityRegularization
from keras.layers.core import Flatten,Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import train_test_split, normalize_aa
from sklearn.preprocessing import OneHotEncoder
import sys
import pdb
import itertools

AALETTER = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


AMINO='ARNDCEQGHILKMFPSTWYV'

# HYDROPHILICITY = normalize_aa(HYDROPHILICITY)
# HYDROPHOBICITY = normalize_aa(HYDROPHOBICITY)
# RESIDUEMASS = normalize_aa(RESIDUEMASS)
# PK1 = normalize_aa(PK1)
# PK2 = normalize_aa(PK2)
# PI = normalize_aa(PI)

LAMBDA = 24
DATA_ROOT = 'level_1/'


def shuffle(*args, **kwargs):
    seed = None
    if 'seed' in kwargs:
        seed = kwargs['seed']
    rng_state = numpy.random.get_state()
    for arg in args:
        if seed is not None:
            numpy.random.seed(seed)
        else:
            numpy.random.set_state(rng_state)
        numpy.random.shuffle(arg)

from fofe_embedding import fofe_features

def load_data(go_id,gram,alpha):
 
    alpha=0.999
    gram = 3
    input_file=DATA_ROOT + go_id + '.txt'
    fofe_data_features, labels = fofe_features(input_file,gram,alpha)
    return fofe_data_features, labels

def model(labels, data):
    # set parameters:
  
    # Training
    batch_size = 100
    nb_epoch = 100

    train, test = train_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    test_label, test_data = test
    test_label_rep = test_label
    shap=numpy.shape(train_data)

    print('X_train shape: ',shap)
    print('X_test shape: ',test_data.shape)
    model = Sequential()
    model.add(Dense(shap[1], activation='relu', input_dim=shap[1]))
    model.add(Highway())
    model.add(Dense(1,activation='sigmoid'))
    print 'compiling model'
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    print 'running at most 60 epochs'
    checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(train_data, train_label, batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True, show_accuracy=True, 
               validation_split=0.3,callbacks=[checkpointer,earlystopper])

    # # Loading saved weights
    print 'Loading weights'
    model.load_weights('bestmodel.hdf5')
    pred_data = model.predict_classes(test_data, batch_size=batch_size)
    # Saving the model
    tresults = model.evaluate(test_data, test_label,show_accuracy=True)
    print tresults
    return classification_report(list(test_label_rep), pred_data)


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')

def main(*args, **kwargs):
    if len(args) != 2:
        sys.exit('Please provide GO Id')
    go_id = args[1]
    print 'Starting binary classification for ' + go_id
    fofe_data_features, labels = load_data(go_id,gram=3,alpha=0.999)
    report = model(labels, fofe_data_features)
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
