import numpy as np
import re
import itertools
from collections import Counter
import pdb

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def load_data_and_labels(sequences,gram):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Split by words
    x_text = [clean_str(sent) for sent in sequences]
    x_text = [s.split(" ") for s in x_text]
    
    return x_text


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


def load_data(gram_seq,vocabulary,vocabulary_inv):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences_padded = pad_sentences(gram_seq)
  #  vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    return x


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

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
