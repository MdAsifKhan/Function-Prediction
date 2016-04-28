'''Train a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
import pandas as pd
import pdb

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

df_train = pd.read_csv('numerai_training_data.csv')
df_test = pd.read_csv('numerai_tournament_data.csv')

id_d = df_test['t_id'].values

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
pdb.set_trace()
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

pdb.set_trace()
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
pdb.set_trace()
model.add(LSTM(128))  # try using a GRU instead, for fun
pdb.set_trace()
model.add(Dropout(0.5))
pdb.set_trace()
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)