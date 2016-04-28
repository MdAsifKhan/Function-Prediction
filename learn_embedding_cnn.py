# -*- coding: utf-8 -*-

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python learn_embedding_cnn.py

import numpy as np
import utils

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
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

def train_test_split(labels, data, split=0.8, batch_size=32):
    """This function is used to split the labels and data
    Input:
        labels - array of labels
        data - array of data
        split - percentage of the split, default=0.8
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (test_labels, test_data)
    """
    n = len(labels)
    train_n = int((n * split) / batch_size) * batch_size

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    train = (train_labels, train_data)

    test_data = data[train_n:]
    test_labels = labels[train_n:]
    test = (test_labels, test_data)

    return (train, test)

AMINO='ARNDCEQGHILKMFPSTWYV'
gram=2
vocabulary_inv = [''.join(item) for item in itertools.product(AMINO,repeat=gram)]
vocabulary_inv =['<PAD/>'] + vocabulary_inv
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# Load data
print("Loading data...")

DATA_ROOT = 'level_1/'
GO_ID = 'GO:0000988'
input_file = DATA_ROOT+GO_ID+".txt"
sequences=[line.strip().split(' ') for line in open(input_file,'rb')]
labels= [lb[0] for lb in sequences]
sequences=[newd[2:][0] for newd in sequences]
seqdata = list()
for seq in range(0,len(sequences)):
    ss = sequences[seq]
    seq1 = [ss[i:i+gram] for i in range(len(ss)-gram+1)]
    seqdata.append(seq1)

sentences_padded = pad_sentences(seqdata)
x = build_input_data(sentences_padded, vocabulary)
# Shuffle data
y = np.array(labels, dtype="float64")
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

print("Vocabulary Size: {:d}".format(len(vocabulary)))


# Building model
sequence_length = np.shape(x_shuffled)[1]
embedding_dim = 20        
nb_filter = 500
filter_length = [20,5]
pool_length = 10

# Training parameters
batch_size = 64
num_epochs = 100
val_split = 0.1

#Data Split
train, test = train_test_split(y_shuffled, x_shuffled, batch_size=batch_size)

train_label, train_data = train
test_label, test_data = test
test_label_rep = test_label
# model
model = Sequential()
model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length[0],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode ='binary')
# Training model
model.fit(train_data, train_label, batch_size=batch_size,
          nb_epoch=num_epochs, show_accuracy=True,
          validation_split=val_split,callbacks=[checkpointer,earlystopper])

# # Loading saved weights
print 'Loading weights'
model.load_weights('bestmodel.hdf5')

pred_data = model.predict_classes(test_data, batch_size=batch_size)

pdb.set_trace()

weights = model.layers[0].get_weights()[0]

from tsne import bh_sne
wt_2d = bh_sne(np.asarray(weights, dtype=np.float64))

import pylab as plot
plot.figure(figsize=(200,200),dpi=100)
wrds = list(vocabulary)
max_x=np.amax(wt_2d,axis=0)[0]
max_y=np.amax(wt_2d,axis=0)[1]
plot.xlim((-max_x,max_x))
plot.ylim((-max_y,max_y))
plot.scatter(wt_2d[:,0],wt_2d[:,1],20)


for rid in range(0,len(wrds)):
    t_wd = wrds[rid]
    x = wt_2d[rid,0]
    y = wt_2d[rid,1]
    plot.annotate(t_wd,(x,y))
plot.savefig("t_sne_word1")

# Saving the model
tresults = model.evaluate(test_data, test_label,show_accuracy=True)
print tresults
print(classification_report(test_label_rep, pred_data))
