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
from sklearn.feature_extraction.text import TfidfVectorizer

from KaggleWord2VecUtility import KaggleWord2VecUtility
from scipy import sparse
from scipy.sparse import lil_matrix

import theano
import theano.tensor as T

import sys
import time
import pylab
import cPickle
import warnings


def makeFeatureVec(num_features,indice,voc_list,tfidf):

    count = 0
    nwords=indice.shape[0]
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
#    index2word_set = set(model.index2word)
    # Pre-initialize an empty numpy array (for speed)           
    featureVec = np.zeros((nwords, num_features),dtype=theano.config.floatX)
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, append the list
#    print tfidf.shape
#    nwords = 0
    tfidf=tfidf[0]
    sumtfidf=0
#    len= indice.shape[0]
    for ind in indice:
#	print tfidf
	sumtfidf+=tfidf[ind]
#	print "tfidf[ind]="
#	print tfidf[ind]
        featureVec[count,:] = voc_list[ind]*tfidf[ind]
        count = count + 1
#    print "sumtfidf=%f" %sumtfidf
#    print featureVec
#    print featureVec.shape
    value=nwords/sumtfidf
#    featureVec=featureVec.sum(axis=0)
#    print "value=%f" %value
    featureVec=featureVec*value
    return featureVec

def getUnidimFeatureVecs(featureVec, uni_num_lines, num_features):
    #warnings.simplefilter("error")

    unidimFeatureVec = np.zeros((uni_num_lines, num_features),dtype=theano.config.floatX)

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

def getWholeFeatureVecs( model, file, num_features, uni_num_lines):
    # Read data from files
    dataset = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', file), header=0, delimiter="\t", quoting=3 )
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    clean_words_list=[]
    #size = []
    counter=0
    index2word_set = set(model.index2word)
    num_reviews= len(dataset["review"])
    #4D matrix
#    wholeFeatureVecs = np.zeros((num_reviews, 1, uni_num_lines, num_features),dtype=theano.config.floatX)

    for review in dataset["review"]:
        #size.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ).__len__())
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
        clean_words_list.append(' '.join(words))

    #remove all the words not in word2vec model
    train_reviews=[]
    for w in clean_words_list:
#       print "split string to list of words"
        words=w.split()
#       print "before filtering, we have %d words in a review" %len(words)
        word=[]
        for w in words:
            if w in index2word_set:
                word.append(w)
#               words.remove(w) 
#       print "1.after filtering,we have %d words in a review" %len(words)
#        print "2.after filtering,we have %d words in a review" %len(word)
        train_reviews.append(" ".join(word))

    vectorizer = TfidfVectorizer(max_df=0.5)
    tfidfFeatureVecs = vectorizer.fit_transform(train_reviews)
#    print "tfidfFeatureVecs[0,0]="
#    print tfidfFeatureVecs[0,0]
    num_reviews=tfidfFeatureVecs.shape[0]
    num_words=tfidfFeatureVecs.shape[1]
#    print "number of words=%d" %num_words

    voc_list=[]
    for i in xrange(0,num_words):
        word=vectorizer.get_feature_names()[i]
        voc_list.append(model[word])
#    print "length of voc_list=%d" %len(voc_list)

    wholeFeatureVecs = np.zeros((num_reviews, 1, uni_num_lines, num_features),dtype="float32")
    cnt=0.
    for i in xrange(0,num_reviews):
	indice=np.nonzero(tfidfFeatureVecs[i].toarray())
#	print indice
#	print indice[1].shape
	indice=indice[1]
	if indice.shape[0]<=uni_num_lines:
	    cnt+=1.
#	print indice
#	print tfidfFeatureVecs[i,indice[0]]
	featureVec=makeFeatureVec(num_features,indice,voc_list,tfidfFeatureVecs[i].toarray())	
	featureVec=getUnidimFeatureVecs(featureVec, uni_num_lines, num_features)
	wholeFeatureVecs[i,:, :, :]=featureVec
    
    path=os.path.join(os.path.dirname(__file__))
    write_file=open(path+'90wholeFeatureVecs','wb')
    for i in xrange(0,wholeFeatureVecs.shape[0]):
        cPickle.dump(wholeFeatureVecs[i],write_file,-1)
    write_file.close()
    print "percentage of indice.shape[0]<=uni_num_lines=%f" %(cnt/num_reviews)
#    wholeFeatureVecs=wholeFeatureVecs.astype('float32')
    print "wholeFeatureVecs generated"
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

    wholeFeatureVecs=getWholeFeatureVecs( model, 'train_labeled_review.tsv', 300, 90)
    print wholeFeatureVecs.shape

