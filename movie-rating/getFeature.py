#!/usr/bin/env python

#  Author: Mina HE
#  Date: 08/09/2015
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
from sklearn.feature_selection import SelectKBest, f_classif,chi2
from KaggleWord2VecUtility import KaggleWord2VecUtility
from scipy import sparse
from scipy.sparse import lil_matrix

import theano
import theano.tensor as T
import gc
import sys
import time

import pylab

import warnings




def makeFeatureVec(words, model, num_features):

#    nwords = 0
    wordlist=[]
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    #construction of wordlist for each review which contains all words in the model and in the review without repeat 
    for word in words:
        if word in index2word_set:
	    if word not in wordlist:
                wordlist.append(word)
    nwords=len(wordlist)        
#    print "len(wordlist)=%d" %nwords
    # Pre-initialize an empty numpy array (for speed)           
    featureVec = sparse.lil_matrix((nwords, num_features),dtype="float32")

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, append the list
    nwords = 0
    for word in wordlist:           
        featureVec[nwords] = model[word]
        nwords = nwords + 1

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
    
    unidimFeatureVec = sparse.lil_matrix((uni_num_lines, num_features),dtype="float32")
    
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

    return unidimFeatureVec.todense()


#whole data set
#file : string
def getWholeFeatureVecs( model, file, num_features, uni_num_lines):
    # Read data from files
    dataset = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', file), header=0, delimiter="\t", quoting=3 )
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    clean_train_reviews=[]
    #size = []
    counter=0
    index2word_set = set(model.index2word)

    num_reviews= len(dataset["review"])
    #4D matrix
    wholeFeatureVecs = np.zeros((num_reviews, 1, uni_num_lines, num_features),dtype="float32")
#    wholeFeatureVecs=[]

    for review in dataset["review"]:
        #size.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ).__len__())
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True )
	clean_train_reviews.append(" ".join(words))        
	
    train_reviews=[]
    for w in clean_train_reviews:
#       print "split string to list of words"
        words=w.split()
	for w in words:
	    if w not in index2word_set:
		words.remove(w) 
	train_reviews.append(" ".join(words))

    vectorizer = TfidfVectorizer(max_df=0.5)
    trainFeatureVecs = vectorizer.fit_transform(train_reviews)

    trainSentimentVecs=getSentimentVecs( model, file )

    train_FeatureVecs=trainFeatureVecs[0:20000]
    train_SentimentVecs=trainSentimentVecs[0:20000]

    valid_FeatureVecs=trainFeatureVecs[20000:25000]
    valid_SentimentVecs=trainSentimentVecs[20000:25000]
    del trainFeatureVecs
    gc.collect()

    selecter=SelectKBest(chi2, k=uni_num_lines)
    train_FeatureVecs=selecter.fit_transform(train_FeatureVecs,train_SentimentVecs)
    valid_FeatureVecs=selecter.transform(valid_FeatureVecs)
#    print "selecter.get_support(True)="
    indice_list= selecter.get_support(True)
#    print indice_list
  
    #construct a list of keywords where each word is one of the k most influential words in all the reviews, at the same time, 
# we can find the word in the word2vec model set
    keywords=[]
    for ind in indice_list:
	word=vectorizer.get_feature_names()[ind]
	if word not in keywords:
	    keywords.append(word)
    print "keywords="
    print keywords
#    len_keywords=len(keywords)
#    print len_keywords

    print "for each review, create a new feature vec only for the keywords"
#    trainFeatureVecs=[]
#    np.set_printoptions(threshold=np.nan)
    #for each review
    counter=0
    for w in train_reviews:
#	print "w in wordsList, generate a zeros matrix"
	#generate a matrix, each keyword a row 
	FeatureVec=sparse.lil_matrix((uni_num_lines,num_features),dtype="float32")
#	print "split string to list of words"
	words=w.split()
#	print words
	for word in words:
#	    print "word in words"
	    if word in keywords:
#		print "if word in keywords"
		row_ind=keywords.index(word)
#		print "row_ind=%d" %row_ind
#		print model[word]
		FeatureVec[row_ind,:]=model[word]
#		print "FeatureVec[row_ind,:]="
#		print FeatureVec[row_ind,:]
#        unidimFeatureVec=getUnidimFeatureVecs(FeatureVec, uni_num_lines, num_features).reshape(1, 1, uni_num_lines, num_features)
        wholeFeatureVecs[counter,:, :, :]=FeatureVec.todense().reshape(1, 1, uni_num_lines, num_features)

        counter += 1
        if counter%1000. == 0.:
           print counter
    print "Parsing end"
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

