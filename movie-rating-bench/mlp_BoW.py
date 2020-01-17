""" 
Author: Mina HE
modified from the version MNIST

Bag of Words combined with mlp

"""
import os
import sys
import time

import pylab

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#import theano.config as config

from logistic_sgd import LogisticRegression, load_data
from mlp import MLP
#from newmlp import MLPClassifier
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif,chi2

from KaggleWord2VecUtility import KaggleWord2VecUtility
""" 
def select_kbest_for_each_review(k,FeatureVecs):
    for i in xrange(0,len(FeatureVecs)):
        list=sorted(FeatureVecs[i],reverse = True)
        threshold=list[k-1]
	if threshold >0:
	    for j in xrange(0,len(FeatureVecs[i])):
	        if FeatureVecs[i][j]<threshold:
		    FeatureVecs[i][j]=0
    return FeatureVecs
"""
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.000, n_epochs=1000,feature_r=95,feature_c=105,
             trainset='bench_labeled_review.tsv',testset='testData.tsv', batch_size=20, n_hidden=500):
    """ 
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
    This is demonstrated on MNIST.
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient
    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)
    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset:
    """
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', trainset), header=0, \
                    delimiter="\t", quoting=3)
#    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', testset), header=0, delimiter="\t", \
#                   quoting=3 )


    rng = np.random.RandomState(23455)

    # print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    # nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

#    print len(train["review"])
#    print "Cleaning and parsing the training set movie reviews...\n"

    for i in xrange( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
    
#    clean_validate_reviews=clean_train_reviews[20000:25000]
#    clean_validate_reviews_df = pd.DataFrame( data={"id":train["id"][20000:25000],"review":train["review"][20000:25000], "clean_review":clean_validate_reviews} )
#    clean_validate_reviews_df.to_csv( "clean_validate_reviews.csv", index=False )
#    print 'Wrote clean_validate_reviews.csv'

    # ****** Create a bag of words from the training set
    #
#    print "Creating the bag of words...\n"
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
#                             max_features = feature_r*feature_c)
     

    vectorizer = TfidfVectorizer(max_df=0.5,ngram_range=(1, 5))
  
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
#    FeatureVecs = vectorizer.fit_transform(clean_train_reviews)
#    print "length of 1-5gram= %d" %len(vectorizer)
    temp=TfidfVectorizer(max_df=0.5).fit_transform(clean_train_reviews)
    print "length of 1-gram= " 
    print temp.shape
    temp=TfidfVectorizer(max_df=0.5,ngram_range=(1, 2)).fit_transform(clean_train_reviews)
    print "length of 1-2gram= " 
    print temp.shape
    temp=TfidfVectorizer(max_df=0.5,ngram_range=(1, 3)).fit_transform(clean_train_reviews)
    print "length of 1-3gram= " 
    print temp.shape
    temp=TfidfVectorizer(max_df=0.5,ngram_range=(1, 4)).fit_transform(clean_train_reviews)
    print "length of 1-4gram= " 
    print temp.shape
    FeatureVecs = vectorizer.fit_transform(clean_train_reviews)
    print "length of 1-5gram= " 
    print FeatureVecs.shape
#    print "len(vectorizer.get_feature_names()) = %s" %len(vectorizer.get_feature_names())
#    for i in xrange(0,30):
#	print vectorizer.get_feature_names()[i]
    # Numpy arrays are easy to work with, so convert the result to an
    # array
#    FeatureVecs = FeatureVecs.toarray()
#    print len(FeatureVecs)

    num_reviews= len(train["sentiment"])
    #1D matrix
    SentimentVecs = np.zeros((num_reviews),dtype="int32")
    counter=0
    for sentiment in train["sentiment"]:
        SentimentVecs[counter]= sentiment
        counter += 1

#    selector = SelectKBest(chi2, k=feature_r*feature_c)
#    selector.fit(FeatureVecs,SentimentVecs)
#    FeatureVecs = selector.transform(FeatureVecs)

    FeatureVecs=SelectKBest(chi2, k=feature_r*feature_c).fit_transform(FeatureVecs,SentimentVecs)
#    for i in xrange(0,30):
#        print FeatureVecs.get_feature_names()[i]

    """ 
    print "Selecting k best features for each review..."
    FeatureVecs=select_kbest_for_each_review(kbest,FeatureVecs)
    tmp=np.sum(FeatureVecs,axis=0)
    print tmp.shape
    print tmp
    c=np.where(tmp==0)
    for ind in c[::-1]:
	FeatureVecs=np.delete(FeatureVecs,ind,1)
    print "FeatureVecs shape"
    print FeatureVecs.shape
    """
    FeatureVecs = FeatureVecs.toarray()
    FeatureVecs = FeatureVecs.reshape(2000, 1, feature_r, feature_c)
    #print trainFeatureVecs[5, 0, 0, ]
    FeatureVecs = FeatureVecs.astype('float32')
    
    trainFeatureVecs, validFeatureVecs, trainSentimentVecs, validSentimentVecs = cross_validation.train_test_split( FeatureVecs,
 SentimentVecs, test_size=0.3, random_state=0)
#    trainFeatureVecs=FeatureVecs[0:1600]
#    trainSentimentVecs=SentimentVecs[0:1600]

#    validFeatureVecs=FeatureVecs[1600:2000]
#    validSentimentVecs=SentimentVecs[1600:2000]

    
    #theano variable
    train_set_x = theano.shared(value=trainFeatureVecs, borrow=True) #20000, 1, feature_r, feature_c
    train_set_y = theano.shared(trainSentimentVecs) #20000

    valid_set_x = theano.shared(value=validFeatureVecs, borrow=True) #5000, 1, feature_r, feature_c
    valid_set_y = theano.shared(validSentimentVecs) #5000

    #theano variable
    #test_set_x = theano.shared(testFeatureVecs)
    test_set_x = train_set_x
    #print test_set_x.shape.eval()
    test_set_y = train_set_y


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches = n_train_batches
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    #x = T.matrix('x')   # the data is presented as rasterized images
    x = T.tensor4('x') 
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
                         

    ######################
    # BUILD ACTUAL MODEL #
    ######################
#    print '... building the model'

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x.flatten(2),
        n_in=feature_r * feature_c,
        n_hidden=n_hidden,
        n_out=2
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5
    ###############
    # TRAIN MODEL #
    ###############
#    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
#    print "validation_frequency="
#    print validation_frequency

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [np.mean(validate_model(i)) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
		""" 
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
		"""
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
 
                    # test it on the test set
#                    test_losses = [test_model(i) for i
#                                   in xrange(n_test_batches)]
#                    test_score = numpy.mean(test_losses)

#                    print(('     epoch %i, minibatch %i/%i, test error of '
#                           'best model %f %%') %
#                          (epoch, minibatch_index + 1, n_train_batches,
#                           test_score * 100.))

            if patience <= iter:
#		print	"patience <= iter"
                done_looping = True
                break

    validate_error = []
    for i in xrange(n_valid_batches):
        validate_error.extend(validate_model(i))

    print 'validate_error=%f' %np.mean(validate_error)

    train_set_error = []
    for i in xrange(n_train_batches):
#       print i
#       print 'validate_model(i)='
#       print validate_model(i)
        train_set_error.extend(test_model(i))
#        print len(train_set_error)
    print 'train_set_error=%f' %np.mean(train_set_error)

    end_time = time.clock()

#    print 'end_time='
#    print end_time
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
         %(best_validation_loss * 100., best_iter + 1.))
#    output = pd.DataFrame( data={"id":train["id"][1600:2000],"review":train["review"][1600:2000], "error":validate_error} )
#    output.to_csv( "mlp_BoW_error.csv", index=False, quoting=3 )
#    output.to_csv( "mlp_BoW_error.csv", index=False )

#    print "Wrote mlp_BoW_error.csv"
    """ 
    mlp = MLPClassifier()
    scores = cross_validation.cross_val_score(mlp, FeatureVecs, SentimentVecs, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    """
if __name__ == '__main__':
#    param_grid = {'feature_r': [50, 55, 60], 'feature_c': [50,55,60]}
    test_mlp()
    """ 
    start_time=time.clock()
    f_r=[90,95,100,105,110,115,120,125]
    f_c=[90,95,100,105,110,115,120,125]
    for r in f_r:
	for c in f_c:
	    print "for feature r= %s,feature c=%s" %(r, c)
	    test_mlp(feature_r=r,feature_c=c)
    end_time=time.clock()
    print "running time= %f" %(end_time-start_time) 
    """
 
