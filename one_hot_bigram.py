#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_sequence1D.py


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

def load_data(go_id,maxlen,dictn,gram):
    data = list()
    labels = list()

    bgram = [''.join(item) for item in itertools.product(AMINO,repeat=gram)]
    input_file=DATA_ROOT + go_id + '.txt'
    docs=[line.strip().split(' ') for line in open(input_file,'rb')]
    labels= [lb[0] for lb in docs]
    docm=[newd[2:][0] for newd in docs]
#    x = numpy.zeros((len(docm),maxlen-gram+1,dictn*gram))
    x = numpy.zeros((len(docm),maxlen,len(bgram)))
    for doc in range(0,len(docm)):
        d = docm[doc]
        seq = [d[i:i+gram] for i in range(len(d)-gram+1)]
        ind=0
        for word in seq[:maxlen]:
            cha = list(word)
   #         vec =[AALETTER.index(ch) for ch in cha]
   #         vec = vec + numpy.linspace(0,dictn*(gram-1),gram)
   #         x[doc,ind,vec.astype(int)] = 1
            col = bgram.index(word)
            x[doc,ind,col] = 1
            ind += 1
    # Previous was 30
    data=x
    shuffle(data, labels, seed=30)
    return numpy.array(labels,dtype="float32"),numpy.array(data, dtype="float32")

def model(go_id):
    # set parameters:
   
    # Convolution
    filter_length = 20
    nb_filter = 32
    pool_length = 10
    stride = 10

    # LSTM
    lstm_output_size = 96

    # Training
    batch_size = 100
    nb_epoch = 60
    patience = 5

    #Encoding
    maxlen = 500
    dictn = 20
    gram = 2

    labels, data = load_data(go_id,maxlen,dictn,gram)

    train, test = train_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    test_label, test_data = test
    test_label_rep = test_label

    

    shap=numpy.shape(train_data)
    print('X_train shape: ',shap)
    print('X_test shape: ',test_data.shape)
    model = Sequential()
    model.add(Convolution1D(input_dim=shap[2],
                        input_length=shap[1],
                        nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length, stride=stride))
    model.add(Dropout(0.75))
    model.add(LSTM(lstm_output_size, return_sequences=True))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.75))
    model.add(Dense(1000))
    model.add(Dense(1,activation='sigmoid'))
    print 'compiling model'
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    print 'running at most 60 epochs'
    checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, show_accuracy=True, 
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
    report = model(go_id)
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
