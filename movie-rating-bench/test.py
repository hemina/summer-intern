#!/usr/bin/env python

#  Author: Xiaoran Jiang
#  Date: 30/03/2015
#

#
# *************************************** #

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

import theano
import theano.tensor as T

import sys
import time

import pylab

import warnings




def makeFeatureVec(words, model, num_features):

    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            
    # Pre-initialize an empty numpy array (for speed)           
    featureVec = np.zeros((nwords, num_features),dtype="float64")

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, append the list
    nwords = 0.
    for word in words:
        if word in index2word_set:           
            featureVec[nwords] = model[word]
            nwords = nwords + 1.

    #print featureVec

##    # Pre-initialize an empty numpy array (for speed)
##    featureVec = np.zeros((words.__len__(), num_features),dtype="float64")
##
##    for word in words:
##        if word in index2word_set:           
##            featureVec[nwords] = model[word]
##            nwords = nwords + 1.
##        #else:
##            #print "hi"

    return featureVec

def getUnidimFeatureVecs(featureVec, uni_num_lines, num_features):
    #warnings.simplefilter("error")
    
    unidimFeatureVec = np.zeros((uni_num_lines, num_features),dtype="float64")
    
    num_lines=featureVec.shape[0]
    
    if num_lines <= uni_num_lines:
        #zero padding
        unidimFeatureVec[0:num_lines, :] = featureVec       
    else:
        #average by slice
##        (div+1)*a+div*b=num_lines
##        a+b=uni_num_lines
##        div=uni_num_lines/num_lines
##        ==>
        b=(num_lines/uni_num_lines+1)*uni_num_lines - num_lines
        a=uni_num_lines-b
        div=num_lines/uni_num_lines

        #average 
        for idx in xrange(0, a):
            unidimFeatureVec[idx]=featureVec[idx*(div+1):(idx+1)*(div+1)].mean(axis=0)
        #rest
        cursor=a*(div+1)
        for idx in xrange(a, a+b):
            unidimFeatureVec[idx]=featureVec[cursor+(idx-a)*div : cursor+(idx-a+1)*div].mean(axis=0)

    #print unidimFeatureVec

    return unidimFeatureVec


#whole data set
#file : string
def getWholeFeatureVecs( model, file, num_features, uni_num_lines):
    # Read data from files
    dataset = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', file), header=0, delimiter="\t", quoting=3 )
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #size = []
    counter=0

    num_reviews= len(dataset["review"])
    #4D matrix
    wholeFeatureVecs = np.zeros((num_reviews, 1, uni_num_lines, num_features),dtype="float64")
    
    for review in dataset["review"]:
        #size.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ).__len__())
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
        
        featureVec = makeFeatureVec(words, model, num_features)
        unidimFeatureVec=getUnidimFeatureVecs(featureVec, uni_num_lines, num_features).reshape(1, 1, uni_num_lines, num_features)

        wholeFeatureVecs[counter,:, :, :]=unidimFeatureVec
        
        counter += 1
        if counter%1000. == 0.:
           print counter 
    print "Parsing ends"

    return wholeFeatureVecs


def getSentimentVecs( model, file ):
    # Read data from files
    dataset = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', file), header=0, delimiter="\t", quoting=3 )
    counter=0

    num_reviews= len(dataset["sentiment"])
    #1D matrix
    wholeSentimentVecs = np.zeros((num_reviews),dtype="int32")
    
    for sentiment in dataset["sentiment"]:
        wholeSentimentVecs[counter]= sentiment      
        counter += 1

    return wholeSentimentVecs
    

if __name__ == '__main__':
    model_name = "300features_40minwords_10context"
    model = Word2Vec.load(model_name)
    # print type(model)
    # print model.most_similar("dog")
    # print model.most_similar("man")
    # print model.most_similar_cosmul(positive=['king','woman'],negative=['man'])
    # vocab=model.vocab
    # for mid in vocab:
		# print(model[mid])
		# print(mid)

    # print model['flower']
    print model.syn0.shape
    print model['flower'].shape

    sentimentVecs=getSentimentVecs( model, 'labeledTrainData.tsv' )
    print sentimentVecs
    print sentimentVecs.shape

    wholeFeatureVecs=getWholeFeatureVecs( model, 'labeledTrainData.tsv', 300, 10)
    print wholeFeatureVecs.shape

    x = theano.shared(wholeFeatureVecs)
    print type(x)

##    # Read data from files
##    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
##    # Load the punkt tokenizer
##    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
##    
##    sentences = []  # Initialize an empty list of sentences
##    size = []
##    counter=0
##    print "Parsing sentences from training set"
##
##    num_reviews= train["review"].shape
##    print num_reviews
##    
##    for review in train["review"]:
##        size.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ).__len__())
##        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
##        words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
##        
##        featureVec = makeFeatureVec(words, model, 300)
##        unidimFeatureVec=getUnidimFeatureVecs(featureVec, 100, 300)
##        #print unidimFeatureVec.shape
##        #print type(unidimFeatureVec)
##        
##        x = theano.shared(unidimFeatureVec)
##        #print type(x)
##        
##        counter += 1
##        if counter%1000. == 0.:
##           print counter 
##    print "Parsing ends"
##    #print min(size)
##    #print max(size)
##    #print np.mean(size)
