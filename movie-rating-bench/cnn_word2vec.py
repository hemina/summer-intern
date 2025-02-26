"""
Author: Xiaoran Jiang
modified from the version MNIST


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

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
from mlp import HiddenLayer

import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

from test import getWholeFeatureVecs, getSentimentVecs


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(5, 5)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.01, n_epochs=200,
                    trainset='labeledTrainData.tsv',
                    testset='testData.tsv',
                    nkerns=[20, 40], batch_size=100):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    model_name = "300features_40minwords_10context"
    model = Word2Vec.load(model_name)
    #number of features
    num_features=300
    #uniform dimension of each review: dim*num_features
    dim=100

    rng = np.random.RandomState(23455)

    print "parsing training data..."
    #train set, type:ndarray
    FeatureVecs=getWholeFeatureVecs( model, trainset, num_features, dim)
    SentimentVecs=getSentimentVecs( model, trainset )

    trainFeatureVecs=FeatureVecs[0:20000]
    trainSentimentVecs=SentimentVecs[0:20000]

    "parsing validation data..."
    validFeatureVecs=FeatureVecs[20000:25000]
    validSentimentVecs=SentimentVecs[20000:25000]

    #print "parsing test data..."
    #test set, type:ndarray
    #testFeatureVecs=getWholeFeatureVecs( model, testset, num_features, dim)

    #theano variable
    train_set_x = theano.shared(trainFeatureVecs) #20000, 1, 50, 300
    print train_set_x.shape.eval()
    train_set_y = theano.shared(trainSentimentVecs) #20000
    print train_set_y.shape.eval()

    valid_set_x = theano.shared(validFeatureVecs) #5000, 1, 50, 300
    print valid_set_x.shape.eval()
    valid_set_y = theano.shared(validSentimentVecs) #5000
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

    #layer0_input = x.reshape((batch_size, 1, 100, 300))
    layer0_input = x

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (100-10+1 , 300-30+1) = (91, 271)
    # maxpooling reduces this further to (91/10, 271/10) = (9, 27)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 27)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 100, 300),
        filter_shape=(nkerns[0], 1, 10, 30),
        poolsize=(10, 10)
    )

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (9-4+1 , 27-12+1) = (6, 16)
    # maxpooling reduces this further to (6/3, 16/4) = (2, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 2, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 9, 27),
        filter_shape=(nkerns[1], nkerns[0], 4, 12),
        poolsize=(3, 4)
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 2 * 4),
    # or (500, 40 * 9 * 1)  with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 2 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    # params = layer2.params + layer1.params + layer0.params
    params = layer3.params+ layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print 'end'
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
##    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
##                                  # go through this many
##                                  # minibatche before checking the network
##                                  # on the validation set; in this case we
##                                  # check every epoch
##
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
##
    epoch = 0
    done_looping = False
##
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 10 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print cost_ij

            #if (iter + 1) % validation_frequency == 0:
            if (iter + 1) % 50 == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

##                    # test it on the test set
##                    test_losses = [
##                        test_model(i)
##                        for i in xrange(n_test_batches)
##                    ]
##                    test_score = np.mean(test_losses)
##                    print(('     epoch %i, minibatch %i/%i, test error of '
##                           'best model %f %%') %
##                          (epoch, minibatch_index + 1, n_train_batches,
##                           test_score * 100.))


                    
##
            if patience <= iter:
                done_looping = True
                break
##
##    end_time = time.clock()
##    print('Optimization complete.')
##    print('Best validation score of %f %% obtained at iteration %i, '
##          'with test performance %f %%' %
##          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
##    print >> sys.stderr, ('The code for file ' +
##                          os.path.split(__file__)[1] +
##                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
