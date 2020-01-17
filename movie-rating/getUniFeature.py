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

import sys
import time
import cPickle
import pylab

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
        words = KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True)
	clean_train_reviews.append(" ".join(words))        
	
    #remove all the words not in word2vec model
    train_reviews=[]
    for w in clean_train_reviews:
#       print "split string to list of words"
        words=w.split()
#	print "before filtering, we have %d words in a review" %len(words)
	word=[]
	for w in words:
	    if w in index2word_set:
		word.append(w)
#		words.remove(w) 
#	print "1.after filtering,we have %d words in a review" %len(words)
#        print "2.after filtering,we have %d words in a review" %len(word)
	train_reviews.append(" ".join(word))

    vectorizer = TfidfVectorizer(max_df=0.5)
    trainFeatureVecs = vectorizer.fit_transform(train_reviews)
    print "trainFeatureVecs.shape="
    print trainFeatureVecs.shape

    num_words=trainFeatureVecs.shape[1]
    voc_list=[]
    for i in xrange(0,num_words):
	word=vectorizer.get_feature_names()[i]
	voc_list.append(model[word])
    print "length of voc_list=%d" %len(voc_list)
    #for each review, select the uni_num_lines words with the highest tfidf
    len_review=trainFeatureVecs.shape[0]
    for i in xrange(0,len_review):
        a=trainFeatureVecs[i,:].toarray()
        featureVec = np.zeros(( uni_num_lines, num_features),dtype="float32")
	k=-uni_num_lines
	indice=a.argsort()[k:][::-1]
#	print "indice="
#	print indice
#	print indice[0]
	cnt=0
	for j in indice[0]:
#	    word=vectorizer.get_feature_names()[j]
#	    print word
	    featureVec[cnt,:]=voc_list[j]
	    cnt+=1
	    if cnt==uni_num_lines:
		break
        if i%1000.==0.:
            print i
        wholeFeatureVecs[i,:, :, :]=featureVec.reshape(1, 1, uni_num_lines, num_features)

    """ 
    for i in xrange(0,trainFeatureVecs.shape[0]):
	a=trainFeatureVecs[i,:].toarray()
        list=sorted(a[0],reverse = True)
        threshold=list[uni_num_lines]
#	print "threshold=%f" %threshold
	cnt=0
#	temp=[]
	featureVec = np.zeros(( uni_num_lines, num_features),dtype="float32")
	#select all the words with a tfidf higher than the threshold
	for j in xrange(0,trainFeatureVecs.shape[1]):
	    if trainFeatureVecs[i,j]>threshold:
#		print "trainFeatureVecs[i,j]>threshold"
#		print "trainFeatureVecs[i,j]=%f" %trainFeatureVecs[i,j]
		word=vectorizer.get_feature_names()[j]
#		print word
		featureVec[cnt,:]=model[word]
		cnt+=1
	#if we have to add some words with their tfidf equal to the threshold
	if cnt<uni_num_lines:
#	    print "cnt<uni_num_lines"
	    for j in xrange(0,trainFeatureVecs.shape[1]):
		if trainFeatureVecs[i,j]==threshold:
		    word=vectorizer.get_feature_names()[j]
                    featureVec[cnt,:]=model[word]
                    cnt+=1
#		    print "cnt=%d" %cnt
		if cnt==uni_num_lines:
		    break
	
	if i%1000.==0.:
	    print i	 
	wholeFeatureVecs[i,:, :, :]=featureVec.reshape(1, 1, uni_num_lines, num_features)
        """
#    print cnt
    print "wholeFeatureVecs[1].shape="
    print wholeFeatureVecs[1].shape

    path=os.path.join(os.path.dirname(__file__))
    write_file=open(path+'wholeFeatureVecs','wb')
    for i in xrange(0,wholeFeatureVecs.shape[0]):
        cPickle.dump(wholeFeatureVecs[i],write_file,-1)
    write_file.close()
    print "wholeFeatureVecs written"
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

#    print model['consult'].shape
#    print model.syn0.shape
#    print model['flower'].shape
    """ 
    sentimentVecs=getSentimentVecs( model, 'bench_labeled_review.tsv' )
    print sentimentVecs
    print sentimentVecs.shape
    """
    wholeFeatureVecs=getWholeFeatureVecs( model, 'train_labeled_review.tsv', 300, 100)
    print wholeFeatureVecs.shape

#    x = theano.shared(wholeFeatureVecs)
#    print type(x)

