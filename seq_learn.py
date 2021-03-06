# -*- coding: utf-8 -*-

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python seq_cnn.py

import numpy as np
import utils

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from sklearn.metrics import classification_report
from keras.utils import np_utils
import pdb
from os.path import join, exists, split
import os
import itertools
import matplotlib.pyplot as plt
np.random.seed(2)


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_input_data(sentences,vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x

def train_test_split(data,target, split=0.15):
    """This function is used to split the labels and data
    Input:
        target - numpy array of Target Data
        data - numpy array of Actual Data
        split - percentage of the split, default=0.15
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (test_labels, test_data)
    """
    
    train_n = round(data.shape[0]*(1-split))
    perms = np.random.permutation(data.shape[0])


    train_data, train_target = data.take(perms[0:train_n],axis=0), target.take(perms[0:train_n],axis=0)

    test_data, test_target = data.take(perms[train_n:],axis=0), target.take(perms[train_n:],axis=0)

    return (train_data, train_target), (test_data, test_target)

AMINO='ARNDCEQGHILKMFPSTWYV'
gram=1
padding_word="<PAD/>"
vocabulary_inv = [''.join(item) for item in itertools.product(AMINO,repeat=gram)]
vocabulary_inv =[padding_word] + vocabulary_inv
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# Load data
print("Loading data...")

DATA_ROOT = 'swissprot/'
GO_ID = 'uniprot-swiss.tab'
input_file = DATA_ROOT+GO_ID
samples=[line.strip().split('\t') for line in open(input_file,'rb')]
sequences=[samples[seq+1][1] for seq in range(0,len(samples)-1)]
#sequences=[newd[2:][0] for newd in sequences]

skip = ['U','O','B','Z','J','X']
seqdata = list()
maxlen = 300
data_x = np.zeros((len(sequences),maxlen,20**gram+1))
data_y = np.zeros((len(sequences),maxlen,20**gram+1))
new_seq = list()
for seq in range(0,len(sequences)):
    ss = sequences[seq]
    seq1 = [ss[i:i+gram] for i in range(len(ss)-gram+1)]
    if len(seq1)<maxlen:
        num_padding = maxlen - len(seq1)
        new_seq = seq1 + [padding_word] * num_padding
    else:
        new_seq = seq1[:maxlen]
    seqdata.append(new_seq)
for i,seq in enumerate(seqdata):
    if any(i in seq for i in skip):
        continue
    else:   
        for t, ng in enumerate(seq):
            data_x[i,t, vocabulary[ng]] = 1
            k = t+1
            if k<len(seq):
                data_y[i,t, vocabulary[seq[k]]] = 1
# Shuffle data
print("Vocabulary Size: {:d}".format(len(vocabulary)))

# Training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

#Data Split

(X_train, y_train), (X_test, y_test) = train_test_split(data_x, data_y, split=0.15)
pdb.set_trace()
# model
shap = np.shape(X_train)
nm_input = shap[2]
n_samples = shap[0]
n_timesteps = shap[1]

shap1 = np.shape(y_train)
nm_output = shap1[2]
nm_hidden = 128
model = Sequential()
model.add(LSTM(nm_hidden,input_dim=nm_input, return_sequences=True))
model.add(LSTM(nm_hidden, return_sequences=True))
model.add(TimeDistributedDense(nm_output))

checkpointer = ModelCheckpoint(filepath="bestmodel_swissprot.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.compile(loss='mse', optimizer='rmsprop')
# Training model

model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=num_epochs,validation_split=val_split,callbacks=[checkpointer,earlystopper])

# # Loading saved weights
print 'Loading weights'
model.load_weights('bestmodel_swissprot.hdf5')



predicted = model.predict(X_test)

pdb.set_trace()
#Plotting Root Mean Square Error (RMSE)
plt.plot(np.sqrt(((predicted-y_test)**2).mean(axis=0)).mean())
plt.show()
