# -*- coding: utf-8 -*-

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python learn_embedding_cnn.py

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

DATA_ROOT = 'level_1/'
go_id = 'GO:0000988.txt'
input_file = DATA_ROOT+go_id
sequences=[line.strip().split(' ') for line in open(input_file,'rb')]
labels= [lb[0] for lb in sequences]
sequences=[newd[2:][0] for newd in sequences]
seqdata = list()
maxlen = 700
data_x = np.zeros((len(sequences),maxlen,20**gram+1))
data_y = np.zeros((len(sequences),maxlen,20**gram+1))
for seq in range(0,len(sequences)):
    ss = sequences[seq]
    seq1 = [ss[i:i+gram] for i in range(len(ss)-gram+1)]
    if len(seq1)<maxlen:
        num_padding = maxlen - len(seq1)
        new_seq = seq1 + [padding_word] * num_padding
    else:
        seq1 = seq1[:maxlen]
    seqdata.append(new_seq)
for i,seq in enumerate(seqdata):
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

# model
shap = np.shape(data_x)
nm_input = shap[2]
n_timesteps = shap[1]
nm_output = nm_input
nm_hidden = 512
model = Sequential()
model.add(LSTM(nm_hidden,input_dim=nm_input, return_sequences=True))
model.add(TimeDistributedDense(nm_output))
model.compile(loss='mse', optimizer='rmsprop')
# Loading saved weights
print 'Loading weights'
model.load_weights('bestmodel.hdf5')
import theano 
get_features = theano.function([model.layers[0].input],model.layers[1].get_output(train=False), allow_input_downcast=True)
data = get_features(data_x)
import utils
labels = np.array(labels,dtype="float32")
from sklearn import cross_validation
train_data,test_data,train_label,test_label = cross_validation.train_test_split(data,labels,test_size=0.3, random_state=42)
test_label_rep = test_label

# Convolution
filter_length = 20
nb_filter = 32
pool_length = 10
stride = 10
patience = 5
nb_epoch = 100

shap=np.shape(train_data)
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
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(1,activation='sigmoid'))
print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
# print('running at most'+ nb_epoch + 'epochs')

checkpointer = ModelCheckpoint(filepath='bestmodel_'+go_id+'.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, show_accuracy=True, 
            validation_split=0.3,callbacks=[checkpointer,earlystopper])


# Loading saved weights
print 'Loading weights'
model.load_weights('bestmodel_'+go_id+'.hdf5')
pred_data = model.predict_classes(test_data, batch_size=batch_size)
# Saving the model
tresults = model.evaluate(test_data, test_label,show_accuracy=True)
print tresults
print(classification_report(list(test_label_rep), pred_data))



