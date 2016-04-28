#!/usr/bin/env python

# FOFE representation

import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

import word_processing
from fofe.fofe import FofeVectorizer
import pdb
# Load data

#DATA_ROOT = 'uniprot/'
#file = 'uniparc-all.tab'

def fofe_features(file_id,gram,alpha):
	print("Loading data...")
	samples=[line.strip().split(' ') for line in open(file_id,'rb')]
	sequences=[samples[seq][2] for seq in range(0,len(samples))]
	labels = [samples[seq][0] for seq in range(0,len(samples))]
	seqdata = list()

	for seq in range(0,len(sequences)):
		dd = sequences[seq]
		seq = [dd[i:i+gram] for i in range(len(dd)-gram+1)]
		seqdata.append(seq)
	
	fofe = FofeVectorizer( alpha )
	x, vocabulary, vocabulary_inv = word_processing.load_data(seqdata)

	print "Vectorizing train (FOFE)..."

	fofe_data_features = fofe.transform( seqdata, vocabulary)

	return fofe_data_features, labels



