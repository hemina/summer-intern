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
import gc
import pylab
import cPickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#import theano.config as config

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from buildmodel import buildWord2vecModel
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

#from dataTrans import getWholeFeatureVecs, getSentimentVecs, getPartialFeatureVecs, getPartialSentimentVecs
from getUniFeature import getWholeFeatureVecs, getSentimentVecs

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

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr = (self.W ** 2).sum()

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(file_object,learning_rate=0.01, n_epochs=100,L2_reg=0.003,
                    trainset='train_labeled_review.tsv',
                    testset='test_labeled_review.tsv',
                    unlabeledset='unlabeled_review.tsv',
		    num_features=300,dim=100,min_word_count=40,
                    nkerns=100, filter_r=5, pool=6, batch_size=50):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

#    buildWord2vecModel(trainset=trainset,testset=testset,unlabeledset=unlabeledset,num_features=num_features,min_word_count=min_word_count)
    model_name = "%dfeatures_%dminwords_10context" %(num_features,min_word_count)
    model = Word2Vec.load(model_name)
    #number of features
#    num_features=200
    #uniform dimension of each review: dim*num_features
#    dim=100

    rng = np.random.RandomState(23455)

    print "parsing training data..."
    #train set, type:ndarray
    featureFilename='wholeFeatureVecs'
#    featureFilename="%dfeatures%ddimWholeFeatureVecs" %(num_features,dim)
    if os.path.exists(featureFilename):
	print "exists('wholeFeatureVecs')"
	FeatureVecs=np.zeros((25000, 1, dim, num_features),dtype="float32")
	path=os.path.join(os.path.dirname(__file__))
        readfile = open(path+featureFilename, 'rb')
        for i in xrange(0,FeatureVecs.shape[0]):
            FeatureVecs[i]=cPickle.load(readfile)
	readfile.close()
    else:
        FeatureVecs=getWholeFeatureVecs( model, trainset, num_features, dim)
    SentimentVecs=getSentimentVecs( model, trainset )
    del model

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
    print "shuffle data..."
    shuffle_in_unison(FeatureVecs, SentimentVecs)

    trainFeatureVecs=FeatureVecs[0:20000]
    trainSentimentVecs=SentimentVecs[0:20000]

    print "parsing validation data..."
    validFeatureVecs=FeatureVecs[20000:25000]
    validSentimentVecs=SentimentVecs[20000:25000]
    del FeatureVecs
    del SentimentVecs
    gc.collect()

##    print "parsing unlabeled data..."
##    unlabeledFeatureVecs=getWholeFeatureVecs( model, unlabeledset, num_features, dim)
##    unlabeledFeatureVecs=unlabeledFeatureVecs[0:5000]
##    unlabeled_set_x = theano.shared(unlabeledFeatureVecs, borrow=True) #50000, 1, 50, 300
##    print unlabeled_set_x.shape.eval()
   
    #theano variable
    train_set_x = theano.shared(trainFeatureVecs, borrow=True) #20000, 1, 50, 300
#    print train_set_x.shape.eval()
    train_set_y = theano.shared(trainSentimentVecs, borrow=True) #20000
#    print train_set_y.shape.eval()
    del trainFeatureVecs
    del trainSentimentVecs
    gc.collect()

    valid_set_x = theano.shared(validFeatureVecs, borrow=True) #5000, 1, 50, 300
#    print valid_set_x.shape.eval()
    valid_set_y = theano.shared(validSentimentVecs, borrow=True) #5000
#    print valid_set_y.shape.eval()
    del validFeatureVecs
    del validSentimentVecs
    gc.collect()

    print "parsing test data..."
##    #test set, type:ndarray
##    testFeatureVecs=getWholeFeatureVecs( model, testset, num_features, dim)
##    #theano variable
##    test_set_x = theano.shared(testFeatureVecs, borrow=True)
    test_set_x = train_set_x
#    print test_set_x.shape.eval()
    test_set_y = train_set_y

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches = n_train_batches
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
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
    print '... building the model'

    #layer0_input = x.reshape((batch_size, 1, 100, 300))
    layer0_input = x

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (100-10+1 , 300-300+1) = (91, 1)
    # maxpooling reduces this further to (91/10, 1/1) = (9, 1)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 1)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, dim, num_features),
        filter_shape=(nkerns, 1, filter_r, num_features),
        poolsize=(pool, 1)
    )

##    # Construct the first convolutional pooling layer:
##    # filtering reduces the image size to (9-4+1 , 27-12+1) = (6, 16)
##    # maxpooling reduces this further to (6/3, 16/4) = (2, 4)
##    # 4D output tensor is thus of shape (batch_size, nkerns[1], 2, 4)
##    layer1 = LeNetConvPoolLayer(
##        rng,
##        input=layer0.output,
##        image_shape=(batch_size, nkerns[0], 9, 27),
##        filter_shape=(nkerns[1], nkerns[0], 4, 12),
##        poolsize=(3, 4)
##    )

    l1feature_r=dim-filter_r+1
    l1feature_c=num_features-num_features+1
    l1feature_r=l1feature_r/pool
    l1feature_c=l1feature_c

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[0] * 16 * 1),
    # or (500, 40 * 9 * 1)  with the default values.
    layer1_input = layer0.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=nkerns * l1feature_r * l1feature_c,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer1.output, n_in=500, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer2.negative_log_likelihood(y)+(layer2.L2_sqr + layer0.L2_sqr)*L2_reg

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer2.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer2.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

 
    

    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params + layer0.params
    # params = layer3.params+ layer2.params + layer1.params + layer0.params

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
    patience_increase = 5  # wait this much longer when a new best is
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

#            if iter % 10 == 0:
#                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
#            print cost_ij

            #if (iter + 1) % validation_frequency == 0:
            if (iter + 1) % 100 == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
		""" 
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
		"""
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
    validate_error = []
    for i in xrange(n_valid_batches):
#       print i
#       print 'validate_model(i)='
#       print validate_model(i)
        validate_error.extend(validate_model(i))
#       print len(validate_error)
    str='validate_error=%f' % np.mean(validate_error)
    print str
    file_object.write(str)

    train_set_error = []
    for i in xrange(n_train_batches):
#       print i
#       print 'validate_model(i)='
#       print validate_model(i)
        train_set_error.extend(test_model(i))
#        print len(train_set_error)
    str= 'train_set_error=%f'%(np.mean(train_set_error))
    print str
    file_object.write(str)




    ##########
##    predict_test_model = theano.function(
##        [index],
##        layer2.y_pred,
##        givens={
##            x: test_set_x[index * batch_size: (index + 1) * batch_size],
##        }
##    )
##
##    #prediction of unlabeled data based on the trained network
##    print('estimation of the labels of test data...')
##    y_pred_list = []
##    for i in xrange(n_test_batches):
##        print('estimation %i/%i' %
##              (i, n_test_batches))
##        #y_pred_list += predict_unlabeled_model(i)
##        y_pred_list.append(predict_test_model(i))
##
##    y_pred_array = np.asarray(y_pred_list)
##    #convert from int64 to int32
##    y_pred_array = y_pred_array.astype(np.int32) #250*100
##    print y_pred_array.shape
##    y_pred_array_flatten = y_pred_array.flatten()
##    #print y_pred_array_flatten[0:20]
##    test_set_y = theano.shared(y_pred_array_flatten, borrow=True) #25000
##    print test_set_y.shape.eval()       





##    def mix_array(a, b):
##       assert len(a) == len(b)
##       c=np.concatenate((a,b), axis=0)
##       p1=np.arange(len(a))
##       p1=p1*2
##       p2=p1+1
##       c[p1]=a
##       c[p2]=b
##       return c
    """ 
    epoch_mixed=0
    n_epochs_mixed=300
    interval=5000
    while (epoch_mixed < n_epochs_mixed):
        epoch_mixed = epoch_mixed + 1
        begin=((epoch_mixed-1)%10)*interval  #10*interval=50000
        end=((epoch_mixed-1)%10+1)*interval      
        print ('epoch %i, parsing unlabeled data from %i to %i...' % (epoch_mixed, begin, end))
        
        unlabeledFeatureVecs=getPartialFeatureVecs( model, unlabeledset, num_features, dim, begin, end)
        unlabeled_set_x = theano.shared(unlabeledFeatureVecs, borrow=True) #50000, 1, 100, 300
        print unlabeled_set_x.shape.eval()

        
        n_unlabeled_batches = unlabeled_set_x.get_value(borrow=True).shape[0]
        n_unlabeled_batches /= batch_size
        print n_unlabeled_batches

        predict_unlabeled_model = theano.function(
            [index],
            layer2.y_pred,
            givens={
                x: unlabeled_set_x[index * batch_size: (index + 1) * batch_size],
            }
        )
    ##
    ##    
        #prediction of unlabeled data based on the trained network
        print('epoch %i, estimation of unlabeled data...' % (epoch_mixed))
        y_pred_list = []
        for i in xrange(n_unlabeled_batches):
#            print('estimation %i/%i' %
#                  (i, n_unlabeled_batches))
            #y_pred_list += predict_unlabeled_model(i)
            y_pred_list.append(predict_unlabeled_model(i))

           
        #y_pred_list = [predict_unlabeled_model(i) for i in xrange(n_unlabeled_batches)]
        y_pred_array = np.asarray(y_pred_list)
        #convert from int64 to int32
        y_pred_array = y_pred_array.astype(np.int32) #500*100
        #print('shuffle the label...')
        #np.random.shuffle(y_pred_array)
        print y_pred_array.shape
        y_pred_array_flatten = y_pred_array.flatten()
    ##    unlabeled_set_y = theano.shared(y_pred_array_flatten, borrow=True) #50000
    ##    print unlabeled_set_y.shape.eval()
    ##
    ##

        print "parsing mixed data (labeled and unlabeled)..."
        begin_labeled=((epoch_mixed-1)%4) *interval  #interval*4=20000
        end_labeled=((epoch_mixed-1)%4 + 1) *interval


##        mixedFeatureVecs=np.concatenate((trainFeatureVecs[begin_labeled:end_labeled], unlabeledFeatureVecs), axis=0)
##        mixedSentimentVecs=np.concatenate((trainSentimentVecs[begin_labeled:end_labeled], y_pred_array_flatten))

##        mixedFeatureVecs=mix_array(trainFeatureVecs[begin_labeled:end_labeled], unlabeledFeatureVecs)
##        mixedSentimentVecs=mix_array(trainSentimentVecs[begin_labeled:end_labeled], y_pred_array_flatten)
        
        mixedFeatureVecs=np.insert(trainFeatureVecs[begin_labeled:end_labeled], np.arange(len(unlabeledFeatureVecs)),unlabeledFeatureVecs, axis=0)
        mixedSentimentVecs=np.insert(trainSentimentVecs[begin_labeled:end_labeled], np.arange(len(y_pred_array_flatten)),y_pred_array_flatten, axis=0)

        del unlabeledFeatureVecs
        del y_pred_array_flatten
	gc.collect()

        #mixedFeatureVecs=np.insert(trainFeatureVecs[begin_labeled:end_labeled],unlabeledFeatureVecsunlabeledFeatureVecs )
            
        #shuffle_in_unison(mixedFeatureVecs, mixedSentimentVecs)

        mixed_set_x=theano.shared(mixedFeatureVecs, borrow=True)
        print mixed_set_x.shape.eval()
        mixed_set_y=theano.shared(mixedSentimentVecs, borrow=True)
        print mixed_set_y.shape.eval()

        n_mixed_batches = mixed_set_x.get_value(borrow=True).shape[0]
        n_mixed_batches /= batch_size
        print n_mixed_batches
        
        
        #training with mixed data (label added)
        train_mixed_model = theano.function(
            [index],
            cost,
            updates=updates, 
            givens={
                x: mixed_set_x[index * batch_size: (index + 1) * batch_size],
                y: mixed_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )


        validation_frequency_bis = min(n_mixed_batches, patience / 2)
    ##                                  # go through this many
    ##                                  # minibatche before checking the network
    ##                                  # on the validation set; in this case we
    ##                                  # check every epoch
    ##
        best_validation_loss_bis = np.inf
        best_iter_bis = 0
        test_score_bis = 0.
    ##

        
        for minibatch_index in xrange(n_mixed_batches):

            iter = (epoch_mixed - 1) * n_mixed_batches + minibatch_index

            if iter % 10 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_mixed_model(minibatch_index)
            print cost_ij
            

            if (iter + 1) % 100 == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch_mixed %i, minibatch %i/%i, validation error %f %%' %
                      (epoch_mixed, minibatch_index + 1, n_mixed_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss_bis:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss_bis *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss_bis = this_validation_loss
                    best_iter_bis = iter
##                   

    """
     
            

if __name__ == '__main__':


##   a=np.array([[0,2],
##              [1,4],
##              [2,5]])
##   b=np.array([[0,3],
##              [3,4],
##              [5,5]])
##   c=np.insert(b, np.arange(len(a)),a, axis=0)
##   print c
##   print a
    start_time = time.clock()
    file_object = open('output_best_word2vec_cnn.txt', 'w')
    """ 
    filter_rs=[5,10,15,20]
    pools=[6,8,10]
    lamdas=[0.001,0.002,0.003]
    num_fs=[200,300]
    dims=[100,150]
    nkernss=[30,50,100]
    min_counts=[30,40]
    for fl_r in filter_rs:
	for p in pools:
	    for ld in lamdas:
		for n_f in num_fs:
		    for di in dims:
			for nk in nkernss:
			    for min_c in min_counts:
				str="\n for filter_r=%d,pool=%d,L2_reg=%f,num_features=%d,dim=%d,nkerns=%d,min_word_count=%d" %(fl_r,p,ld,n_f,di,nk,min_c)
				print str
				file_object.write(str)
				evaluate_lenet5(file_object,L2_reg=ld,num_features=n_f,dim=di,min_word_count=min_c,nkerns=nk, filter_r=fl_r, pool=p)
    """
    evaluate_lenet5(file_object)
    end_time = time.clock()
    print "running time=%f" %(end_time-start_time)
    file_object.close()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
