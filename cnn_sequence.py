#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/cnn_sequence.py


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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import train_val_test_split, normalize_aa
from sklearn.preprocessing import OneHotEncoder
import sys

AALETTER = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


# HYDROPHILICITY = normalize_aa(HYDROPHILICITY)
# HYDROPHOBICITY = normalize_aa(HYDROPHOBICITY)
# RESIDUEMASS = normalize_aa(RESIDUEMASS)
# PK1 = normalize_aa(PK1)
# PK2 = normalize_aa(PK2)
# PI = normalize_aa(PI)

LAMBDA = 24
DATA_ROOT = 'deepfunc-master/level_1/'

encoder=OneHotEncoder()
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

def init_encoder():
    data=list()
    for l in AALETTER:
        data.append([ord(l)])
    encoder.fit(data)
init_encoder()
def encode_seq(seq):
    data=list()
    for l in seq:
        data.append([ord(l)])
    data=encoder.transform(data).toarray()
    return list(data)

def load_data(go_id):
    data = list()
    labels = list()
    pos = 1
    positive = list()
    negative = list()
    with open(DATA_ROOT + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq=[]
            seq=encode_seq(line[2][:500])
            while len(seq)<500:
		seq.append(numpy.zeros(20))
                  
            if label == pos:
                positive.append(seq)
            else:
                negative.append(seq)
    shuffle(negative, seed=10)
    n = len(positive)
    data = negative[:n] + positive
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data, labels, seed=30)
    return numpy.array(labels), numpy.array(data, dtype="float32")


def model(labels, data, go_id):
    # set parameters:
    # Embedding
    # Convolution
    nb_conv = 7
    nb_filter = 64
    nb_pool = 2

  
    # Training
    batch_size = 30
    nb_epoch = 12

    train, val, test = train_val_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    val_label, val_data = val
    test_label, test_data = test
    test_label_rep = test_label
    
    train_data = train_data.reshape(train_data.shape[0], 1, 500, 20)
    test_data = test_data.reshape(test_data.shape[0], 1, 500, 20)
    val_data = val_data.reshape(val_data.shape[0], 1, 500, 20)
    model = Sequential()
    model.add(Convolution2D(96, nb_conv, 1,
                        border_mode='valid',
                        input_shape=(1, 500, 20)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter, 3, 1,
                        border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')

    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_data=(val_data, val_label))
    # # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    pred_data = model.predict_classes(test_data, batch_size=batch_size)
    # Saving the model
    print 'Saving the model for ' + go_id
    model.save_weights(DATA_ROOT + go_id + '.hdf5', overwrite=True)
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
    labels, data = load_data(go_id)
    report = model(labels, data, go_id)
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
