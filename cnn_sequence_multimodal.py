
#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_sequence_multimodal.py
The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
The hydrophilicity values are from PNAS, 1981, 78:3824-3828
(T.P.Hopp & K.R.Woods). The side-chain mass for each of the 20 amino acids. CRC
Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton,
Florida (1985). R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones,
Data for Biochemical Research 3rd ed.,
Clarendon Press Oxford (1986).
'''

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Highway
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from utils import (
    train_test_split, normalize_aa,
    shuffle, encode_seq_one_hot
)
import sys
import pdb
from keras.optimizers import Adam
from aaindex import (
    LUTR910101, NORM_AACOOC, AAINDEX,
    ALTS910101, CROG050101, OGAK980101,
    KOLA920101, TOBD000101, MEHP950102)
from keras.callbacks import ModelCheckpoint, EarlyStopping


MAXLEN = 500
DATA_ROOT = 'level_1/'

def load_data(go_id):
    data1 = list()
    data2 = list()
    labels = list()
    positive1 = list()
    negative1 = list()
    positive2 = list()
    negative2 = list()

    with open(DATA_ROOT + '/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = line[2][:MAXLEN]
            sq1 = encode_seq_one_hot(seq, maxlen=MAXLEN)
            sq2 = list()
            for l in seq:
                sq2.append(MEHP950102[AAINDEX[l]])
            while len(sq2) < MAXLEN:
                sq2.append([0.0] * 20)
            if label == 1:
                positive1.append(sq1)
                positive2.append(sq2)
            else:
                negative1.append(sq1)
                negative2.append(sq2)
    shuffle(negative1, negative2)
    n = min(len(positive1), len(negative1))
    data1 = negative1[:n] + positive1[:n]
    data2 = negative2[:n] + positive2[:n]
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data1, data2, labels)
    return (
        numpy.array(labels),
        numpy.array(data1, dtype='float32'),
        numpy.array(data2, dtype='float32'),
        numpy.array(data1, dtype='float32'))

def model(labels, data1, data2, data3, go_id):
    # set parameters:
    # Convolution
    nb_filter = 64
    nb_row = 3
    nb_col = 1

    pool_length = 2

    # Training
    batch_size = 32
    nb_epoch = 10

    lstm_size = 70

    train1, test1 = train_test_split(
        labels, data1, batch_size=batch_size, split=0.8)
    train_label, train1_data = train1

    train2, test2 = train_test_split(
        labels, data2, batch_size=batch_size, split=0.8)
    train_label, train2_data = train2

    train3, test3 = train_test_split(
        labels, data3, batch_size=batch_size, split=0.8)
    train_label, train3_data = train3

    if len(train1_data) < 100:
        raise Exception("No training data for " + go_id)
    test_label, test1_data = test1
    test_label, test2_data = test2
    test_label, test3_data = test3
    test_label_rep = test_label


    first = Sequential()
    first.add(Convolution1D(input_dim=20,
                        input_length=500,
                        nb_filter=640,
                        filter_length=20,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
    first.add(MaxPooling1D(pool_length=10, stride=10))
    first.add(Dropout(0.2))
    
    second = Sequential()
    second.add(Convolution1D(input_dim=20,
                        input_length=500,
                        nb_filter=640,
                        filter_length=20,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
    second.add(Activation('relu'))
    second.add(MaxPooling1D(pool_length=10, stride=10))
    second.add(Dropout(0.2))

    model = Sequential()
    model.add(Merge([first, second], mode='concat'))
    model.add(LSTM(input_dim=320, output_dim=320, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")


    checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(X=[train1_data, train2_data], y=train_label, batch_size=100, nb_epoch=60, shuffle=True, show_accuracy=True, 
        validation_split=0.3, callbacks=[checkpointer,earlystopper])

    # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    pred_data = model.predict_classes(
        [test1_data, test2_data], batch_size=batch_size)
    # Saving the model
    print 'Saving the model for ' + go_id
    model.save_weights(DATA_ROOT + '/' + go_id + '.hdf5', overwrite=True)
    return classification_report(list(test_label_rep), pred_data)


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + '/' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    if len(args) < 2:
        sys.exit('Please provide GO Id')
    go_id = args[1]
    print 'Starting binary classification for ' + go_id
    labels, data1, data2, data3 = load_data(go_id)
    report = model(labels, data1, data2, data3, go_id)
    print report
    print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
