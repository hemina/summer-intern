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

import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

from test import getWholeFeatureVecs, getSentimentVecs

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.000, n_epochs=1000,
             trainset='labeledTrainData.tsv',testset='testData.tsv', batch_size=20, n_hidden=500):
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
    model_name = "300features_40minwords_10context"
    model = Word2Vec.load(model_name)
    #number of features
    num_features=300
    #uniform dimension of each review: dim*num_features
    dim=200

    rng = np.random.RandomState(23455)

    print "parsing training data..."
    #train set, type:ndarray
    FeatureVecs=getWholeFeatureVecs( model, trainset, num_features, dim)
    FeatureVecs = FeatureVecs.astype('float32')
    SentimentVecs=getSentimentVecs( model, trainset )

    trainFeatureVecs=FeatureVecs[0:1800]
    trainSentimentVecs=SentimentVecs[0:1800]

    "parsing validation data..."
    validFeatureVecs=FeatureVecs[1800:2000]
    validSentimentVecs=SentimentVecs[1800:2000]

    #print "parsing test data..."
    #test set, type:ndarray
    #testFeatureVecs=getWholeFeatureVecs( model, testset, num_features, dim)

    #theano variable
    train_set_x = theano.shared(value=trainFeatureVecs,borrow=True) #20000, 1, 300, 100
    print train_set_x.shape.eval()
    train_set_y = theano.shared(value=trainSentimentVecs,borrow=True) #20000
    print train_set_y.shape.eval()

    valid_set_x = theano.shared(value=validFeatureVecs,borrow=True) #5000, 1, 300, 100
    print valid_set_x.shape.eval()
    valid_set_y = theano.shared(value=validSentimentVecs,borrow=True) #5000
    print valid_set_y.shape.eval()

    #theano variable
    #test_set_x = theano.shared(testFeatureVecs)
    test_set_x = train_set_x
    print test_set_x.shape.eval()
    test_set_y = train_set_y



    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches = n_train_batches
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_valid_batches /= batch_size
    #n_test_batches /= batch_size

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
    print '... building the model'

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x.flatten(2),
        n_in=num_features * dim,
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
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 50  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    print "validation_frequency="
    print validation_frequency

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

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

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
		print	"patience <= iter"
                done_looping = True
                break

    validate_error = []
    for i in xrange(n_valid_batches):
#	print i
#	print 'validate_model(i)='
#	print validate_model(i)
        validate_error.extend(validate_model(i))
#	print len(validate_error)
    print 'validate_error='
    print np.mean(validate_error)

    train_set_error = []
    for i in xrange(n_train_batches):
#       print i
#       print 'validate_model(i)='
#       print validate_model(i)
        train_set_error.extend(test_model(i))
#        print len(train_set_error)
    print 'train_set_error='
    print np.mean(train_set_error)

    end_time = time.clock()

    print 'end_time='
    print end_time
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
         %(best_validation_loss * 100., best_iter + 1.))
#    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', trainset), header=0, \
#                    delimiter="\t", quoting=3)
#    output = pd.DataFrame( data={"id":train["id"][20000:25000],"review":train["review"][20000:25000], "error":validate_error} )
#    output.to_csv( "mlp_BoW_error.csv", index=False, quoting=3 )
#    output.to_csv( "mlp_Word2vec_error.csv", index=False )

    print "Wrote mlp_Word2vec_error.csv"


if __name__ == '__main__':
    test_mlp()

