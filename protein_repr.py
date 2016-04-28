# -*- coding: utf-8 -*-

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python text_embedding.py

import numpy as np
import word_processing 

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from sklearn.metrics import classification_report
from keras.utils import np_utils
import pdb
from gensim.models import word2vec
from sklearn.preprocessing import normalize
from os.path import join, exists, split
import os
import itertools

np.random.seed(2)

# Load data
print("Loading data...")

AMINO='ARNDCEQGHILKMFPSTWYV'
gram=3
vocabulary_inv = [''.join(item) for item in itertools.product(AMINO,repeat=gram)]
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

DATA_ROOT = 'uniprot/'
file = 'uniparc-all.tab'
samples=[line.strip().split('\t') for line in open(DATA_ROOT+file,'rb')]
sequences=[samples[seq+1][1] for seq in range(0,len(samples)-1)]
#seqdata = open("ngram_seq_data.txt",'a')
seqdata = list()
gram=3
for seq in range(0,len(sequences)):
    dd = sequences[seq]
    seq = [dd[i:i+gram] for i in range(len(dd)-gram+1)]
    seqdata.append(seq)
pdb.set_trace()
x = word_processing.load_data(seq,vocabulary,vocabulary_inv)
"""
    seqdata.writelines("%s " % se for se in seq)
    seqdata.write("\n")
seqdata.close()
"""
print("Vocabulary Size: {:d}".format(len(vocabulary)))
pdb.set_trace()
# Word2vec Parameters
num_features = 100        
context = 5

# CNN Parameters
embedding_dim = num_features
nb_filter = 32
filter_length = [3,4]
pool_length = 2
nb_classes = 3

# Training parameters
batch_size = 100
num_epochs = 100
val_split = 0.1

model_dir = 'word2vec_models'
model_name = "{:d}features_{:d}context".format(num_features,context)
model_name = join(model_dir, model_name)

if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print 'Loading existing Word2Vec model \'%s\'' % split(model_name)[-1]
else:
    num_workers = 42       # Number of threads to run in parallel
    downsampling = 1e-3   # Downsample setting for frequent words
    # Initialize and train the model
    print "Training Word2Vec model..."
    docs = [[vocabulary_inv[w] for w in s] for s in x]
    embedding_model = word2vec.Word2Vec(docs,workers=num_workers,size=num_features, min_count = 1,window = context, sample = 1e-3)

    if not exists(model_dir):
        os.mkdir(model_dir)
    print 'Saving Word2Vec model \'%s\'' % split(model_name)[-1]
    embedding_model.save(model_name)
embedding_weights = [np.array([embedding_model[w] if w in embedding_model else np.random.uniform(-0.25,0.25,embedding_model.vector_size) for w in vocabulary_inv])]

embedding_weights = normalize(embedding_weights[0])
shap = np.shape(embedding_weights)  
embedding_weights = embedding_weights.reshape(1,shap[0],shap[1])
"""
import matplotlib.pyplot as plt
weights = embedding_weights[0]
def plot_annotation(wt_2d,annotations):
    plt.figure(figsize=(200,200),dpi=100)
    for r, at in enumerate(annotations):
        x, y = wt_2d[r,:]
        plt.scatter(x,y)
        plt.annotate(at, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
plt.savefig('tsne_w2vec500.png')

from tsne import bh_sne
wt_2d = bh_sne(np.asarray(weights, dtype=np.float64))
points=500
wt_embd = wt_2d[:points,:]
annotations = [vocabulary_inv[i] for i in xrange(points)]
plot_annotation(wt_embd,annotations)

pdb.set_trace()
"""
# Shuffle data
x_test=x[-41:]
x = x[:-41]
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Building model
sequence_length = np.shape(x_shuffled)[1]

#Data Split
train, test = utils.train_test_split(y_shuffled, x_shuffled, batch_size=batch_size)

train_label, train_data = train
test_label, test_data = test
test_label_rep = test_label
# model

model = Sequential()
model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,weights=embedding_weights))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length[0],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# Training model
model.fit(train_data, train_label, batch_size=batch_size,
          nb_epoch=num_epochs, show_accuracy=True,
          validation_split=val_split,callbacks=[checkpointer,earlystopper])

# # Loading saved weights
print 'Loading weights'
model.load_weights('bestmodel.hdf5')

pred_data = model.predict_classes(test_data, batch_size=batch_size)

# Saving the model
tresults = model.evaluate(test_data, test_label,show_accuracy=True)
print tresults
print(classification_report(np.argmax(test_label_rep,axis=1), pred_data))

pred_data = model.predict_classes(x_test, batch_size=batch_size)

proba = model.predict(x_test,verbose=1)
pdb.set_trace()