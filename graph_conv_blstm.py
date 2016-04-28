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

import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Highway, Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import train_test_split, normalize_aa
from sklearn.preprocessing import OneHotEncoder
import sys
import pdb
from os.path import join, exists, split
import os
import itertools

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
DATA_ROOT = 'level_1/'

encoder=OneHotEncoder()
def shuffle(*args, **kwargs):
    seed = None
    if 'seed' in kwargs:
        seed = kwargs['seed']
    rng_state = np.random.get_state()
    for arg in args:
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.set_state(rng_state)
        np.random.shuffle(arg)

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

def model():
    # set parameters:
   
    AMINO='ARNDCEQGHILKMFPSTWYV'
    gram=1
    vocabulary_inv = [''.join(item) for item in itertools.product(AMINO,repeat=gram)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    # Load data
    print("Loading data...")

    DATA_ROOT = 'uniprot-swiss/'
    uniprot_file = 'uniprot-swiss-mol-func-yeast'
    input_file = DATA_ROOT+uniprot_file+".txt"
    
    sequences=[line.strip().split('\t') for line in open(input_file,'rb')]
    sequences= [seq[1] for seq in sequences]
    go_id=[newd[2] for newd in sequences]
    seqdata = list()
    maxlen = 700
    for seq in range(0,len(sequences)):
        ss = sequences[seq]
        seq1 = [ss[i:i+gram] for i in range(len(ss)-gram+1)]
        new_seq = seq1[:maxlen]
        seqdata.append(new_seq)

    data_x = np.zeros((len(sequences),maxlen,20**gram))

    labels = np.zeros((len(sequences),44))
    for i,seq in enumerate(seqdata):
        for t, ng in enumerate(seq):
            data_x[i,t, vocabulary[ng]] = 1

    pdb.set_trace()
    # Shuffle data
    y = np.array(labels, dtype="float64")
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    print("Vocabulary Size: {:d}".format(len(vocabulary)))


    # Convolution
    filter_length = 7
    nb_filter = 64
    pool_length = 2
    k=7
    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    nb_epoch = 12

    train, test = train_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    test_label, test_data = test
    test_label_rep = test_label



    model = Graph()
    model.add_input(name='input',input_shape=((1000,20)))
    model.add_node(Convolution1D(nb_filter=96,
                        filter_length=7,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1),name='conv1', input='input')
    model.add_node(MaxPooling1D(pool_length=3,stride=1),input='conv1', name='pool1')
    model.add_node(Dropout(0.5),input='conv1', name='drop1')


    model.add_node(Dense(100),input='conv1',name='level1')
    model.add_node(Dense(26),input='level1',name='GO:0003674')


    model.add_node(Dense(100),input='G0:0003674',name='level2')
    model.add_node(Dense(7),input='level2', name='G0:0016787')

    model.add_node(Dense(100),input='GO:0003674',name='level2a')
    model.add_node(Dense(4),input='level2a', name='GO:0016740')


    model.add_node(Dense(100),input='GO:0003674',name='level2b')
    model.add_node(Dense(5),input='level2b', name='GO:0003723')

    model.add_node(Dense(100),input='GO:0003674',name='level2c')
    model.add_node(Dense(1),input='level2c', name='GO:0042393')

    model.add_node(Dense(100),input='GO:0003674',name='level2d')
    model.add_node(Dense(1),input='level2d', name='GO:0005085')

    model.add_node(Dense(100),input='GO:0003674',name='level2e')
    model.add_node(Dense(1),input='level2e', name='GO:0019899')

    model.add_node(Dense(100),input='GO:0003674',name='level2f')
    model.add_node(Dense(1),input='level2f', name='GO:0001071')    

    model.add_node(Dense(100),input='GO:0003674',name='level2g')
    model.add_node(Dense(1),input='level2g', name='GO:0004871')


    model.add_node(Dense(100),input='GO:0003674',name='level2h')
    model.add_node(Dense(1),input='level2h', name='GO:0003677')

    model.add_node(Dense(100),input='GO:0003674',name='level2i')
    model.add_node(Dense(1),input='level2i', name='GO:0005198')    

    model.add_node(Dense(100),input='GO:0003674',name='level2j')
    model.add_node(Dense(1),input='level2j', name='GO:0030674')    

    model.add_node(Dense(100),input='GO:0003674',name='level2k')
    model.add_node(Dense(1),input='level2k', name='GO:0000988') 

    model.add_node(Dense(100),input='GO:0003674',name='level2l')
    model.add_node(Dense(1),input='level2l', name='GO:0008092')    

    model.add_node(Dense(100),input='GO:0003674',name='level2m')
    model.add_node(Dense(1),input='level2m', name='GO:0016829')

    model.add_node(Dense(100),input='GO:0003674',name='level2n')
    model.add_node(Dense(1),input='level2n', name='GO:0043167') 

    model.add_node(Dense(100),input='GO:0003674',name='level2o')
    model.add_node(Dense(1),input='level2o', name='GO:0051082')    

    model.add_node(Dense(100),input='GO:0003674',name='level2p')
    model.add_node(Dense(1),input='level2p', name='GO:0016491')    

    model.add_node(Dense(100),input='GO:0003674',name='level2q')
    model.add_node(Dense(1),input='level2q', name='GO:0008134')    

    model.add_node(Dense(100),input='GO:0003674',name='level2r')
    model.add_node(Dense(1),input='level2r', name='GO:0008289') 

    model.add_node(Dense(100),input='GO:0003674',name='level2s')
    model.add_node(Dense(1),input='level2s', name='GO:0016853')    

    model.add_node(Dense(100),input='GO:0003674',name='level2t')
    model.add_node(Dense(1),input='level2t', name='GO:0032182') 

    model.add_node(Dense(100),input='GO:0003674',name='level2u')
    model.add_node(Dense(1),input='level2u', name='GO:0003682')    

    model.add_node(Dense(100),input='GO:0003674',name='level2v')
    model.add_node(Dense(1),input='level2v', name='GO:0008565')    

    model.add_node(Dense(100),input='GO:0003674',name='level2w')
    model.add_node(Dense(1),input='level2w', name='GO:0030234')    

    model.add_node(Dense(100),input='GO:0003674',name='level2x')
    model.add_node(Dense(1),input='level2x', name='GO:0016874')    

    model.add_node(Dense(100),input='GO:0003674',name='level2y')
    model.add_node(Dense(1),input='level2y', name='GO:0022857')    



    model.add_node(Dense(100),input='GO:0016740',name='level3')
    model.add_node(Dense(1),input='level3', name='GO:0016757')

    model.add_node(Dense(100),input='GO:0016740',name='level3a')
    model.add_node(Dense(1),input='level3a', name='GO:0008168')

    model.add_node(Dense(100),input='GO:0016740',name='level3b')
    model.add_node(Dense(1),input='level3b', name='GO:0016301')

    model.add_node(Dense(100),input='GO:0016740',name='level3c')
    model.add_node(Dense(1),input='level3c', name='GO:0016779')



    model.add_node(Dense(100),input='GO:0016787',name='level3d')
    model.add_node(Dense(1),input='level3d', name='GO:0008233')

    model.add_node(Dense(100),input='GO:0016787',name='level3e')
    model.add_node(Dense(1),input='level3e', name='GO:0003924')

    model.add_node(Dense(100),input='GO:0016787',name='level3f')
    model.add_node(Dense(1),input='level3f', name='GO:0016791')

    model.add_node(Dense(100),input='GO:0016787',name='level3g')
    model.add_node(Dense(1),input='level3g', name='GO:0016798')

    model.add_node(Dense(100),input='GO:0016787',name='level3h')
    model.add_node(Dense(1),input='level3h', name='GO:0004386')

    model.add_node(Dense(100),input='GO:0016787',name='level3i')
    model.add_node(Dense(1),input='level3i', name='GO:0016887')

    model.add_node(Dense(100),input='GO:0016787',name='level3j')
    model.add_node(Dense(1),input='level3j', name='GO:0004518')

    
    model.add_node(Dense(100),input='GO:0003723',name='level3k')
    model.add_node(Dense(1),input='level3k', name='GO:0008135')

    model.add_node(Dense(100),input='GO:0003723',name='level3l')
    model.add_node(Dense(1),input='level3l', name='GO:0030555')

    model.add_node(Dense(100),input='GO:0003723',name='level3m')
    model.add_node(Dense(1),input='level3m', name='GO:0003729')

    model.add_node(Dense(100),input='GO:0003723',name='level3n')
    model.add_node(Dense(1),input='level3n', name='GO:0030553')

    model.add_node(Dense(100),input='GO:0003723',name='level3o')
    model.add_node(Dense(1),input='level3o', name='GO:0019843')

    model.add_node(Dense(100),input='GO:0005198',name='level3p')
    model.add_node(Dense(1),input='level3p', name='GO:0005198')


    model.add_node(Dense(44, activation='softmax'), name='owl:nothing', inputs=['GO:0003674',
        'G0:0016787','GO:0016740','GO:0003723','GO:0042393','GO:0005085','GO:0019899','GO:0001071','GO:0004871','GO:0003677','GO:0005198','GO:0030674','GO:0000988','GO:0008092',
        'GO:0016829','GO:0043167','GO:0051082','GO:0016491','GO:0008134','GO:0008289','GO:0016853','GO:0032182','GO:0003682','GO:0008565','GO:0030234','GO:0016874','GO:0022857',
        'GO:0016757','GO:0008168','GO:0016301','GO:0016779',
        'GO:0008233','GO:0003924','GO:0016791','GO:0016798','GO:0004386','GO:0016887','GO:0004518',
        'GO:0008135','GO:0030555','GO:0003729','GO:0030553','GO:0019843',
        'GO:0005198'])

    model.add_output(name='output', input='owl:nothing')
    print 'compiling model'
    model.compile('rmsprop',{'output': 'binary_crossentropy'})
    print 'running at most 60 epochs'
    checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


    model.fit({'input':train_data, 'output':train_label}, batch_size=64, nb_epoch=60, 
        validation_split=0.2, callbacks=[checkpointer,earlystopper])

    # # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    model.load_weights('bestmodel.hdf5')
    pred_data=numpy.round(numpy.array(model.predict({'input': test_data},
                                               batch_size=10)['output']))
    acc = accuracy(test_label,pred_data)
    # Saving the model
    #tresults = model.evaluate(test_data, test_label,show_accuracy=True)
    #print tresults
    print acc
    return classification_report(list(test_label_rep), pred_data)


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')

def get_gene_ontology(filename='go.obo'):
   # Reading Gene Ontology from OBO Formatted file
   go = dict()
   obj = None
   with open(filename, 'r') as f:
       for line in f:
           line = line.strip()
           if not line:
               continue
           if line == '[Term]':
               if obj is not None:
                   go[obj['id']] = obj
               obj = dict()
               obj['is_a'] = list()
               obj['is_obsolete'] = False
               continue
           elif line == '[Typedef]':
               obj = None
           else:
               if obj is None:
                   continue
               l = line.split(": ")
               if l[0] == 'id':
                   obj['id'] = l[1]
               elif l[0] == 'is_a':
                   obj['is_a'].append(l[1].split(' ! ')[0])
               elif l[0] == 'is_obsolete' and l[1] == 'true':
                   obj['is_obsolete'] = True
   if obj is not None:
       go[obj['id']] = obj
   for go_id in go.keys():
       if go[go_id]['is_obsolete']:
           del go[go_id]
   for go_id, val in go.iteritems():
       if 'children' not in val:
           val['children'] = list()
       for g_id in val['is_a']:
           if 'children' not in go[g_id]:
               go[g_id]['children'] = list()
           go[g_id]['children'].append(go_id)
   return go

def main(*args, **kwargs):

    report = model()
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
