#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import cv2
import Constants as c
import theano
from network import Network,ConvPoolLayer,RandCombLayer
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
from theano.tensor import tanh

#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "code.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "code.py to set\nthe GPU flag to True."

def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

class MultiScale(object):

    def __init__(self):
        """Takes a list of `networks`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        filter_shape1=c.filter_shape1
        filter_shape2=c.filter_shape2
        poolsize=c.poolsize
        n_out1=(np.prod(filter_shape1)/np.prod(poolsize))
        n_out2=(np.prod(filter_shape2)/np.prod(poolsize))
        self.wt_conv=[theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out1),size=filter_shape1),dtype=theano.config.floatX),borrow=True),
                      theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out2),size=filter_shape2),dtype=theano.config.floatX),borrow=True)]
        self.b_conv= theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(16)),dtype=theano.config.floatX),borrow=True)
        self.wt64=theano.shared(np.asarray(np.random.normal(loc=0,scale=1.0,size=(16,64)),dtype=theano.config.floatX),borrow=True)
        self.b64=theano.shared(np.asarray(np.random.normal(loc=0,scale=1.0,size=64),dtype=theano.config.floatX),borrow=True)
        self.wt256=theano.shared(np.asarray(np.random.normal(loc=0,scale=1.0,size=(64,256)),dtype=theano.config.floatX),borrow=True)
        self.b256=theano.shared(np.asarray(np.random.normal(loc=0,scale=1.0,size=256),dtype=theano.config.floatX),borrow=True)
        self.input=[T.tensor4("x1"),T.tensor4("x2"),T.tensor4("x3")]
        self.y=T.tensor3("y")
        self.nets=[Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombLayer(self.wt64,self.b64),
            RandCombLayer(self.wt256,self.b256)],self.input[0],c.image_shape1),
                   Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombLayer(self.wt64,self.b64),
            RandCombLayer(self.wt256,self.b256)],self.input[1],c.image_shape2),
                   Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombLayer(self.wt64,self.b64),
            RandCombLayer(self.wt256,self.b256)],self.input[2],c.image_shape3)]

        self.num_class=c.num_class
        self.softmax=SoftmaxLayer(768,self.num_class)
        self.mini_batch_size = c.mini_batch_size
        self.params =[w for w in self.wt_conv]+[self.wt64]+[self.b64]+[self.wt256]+[self.b256] + [self.b_conv]+ self.softmax.params
        image_size=c.image_size
        """output1=skimage.transform.pyramid_expand(self.nets[0].output,upscale=image_size/c.net_out_size[0], sigma=None, order=1, mode='reflect', cval=0)
        output2=skimage.transform.pyramid_expand(self.nets[1].output,upscale=image_size/c.net_out_size[1], sigma=None, order=1, mode='reflect', cval=0)
        output3=skimage.transform.pyramid_expand(self.nets[2].output,upscale=image_size/c.net_out_size[2], sigma=None, order=1, mode='reflect', cval=0)"""
        output1=functions.upsample4D(self.nets[0].output,image_size/c.net_out_size[0],self.mini_batch_size,image_size)
        output2=functions.upsample4D(self.nets[1].output,image_size/c.net_out_size[1],self.mini_batch_size,image_size)
        output3=functions.upsample4D(self.nets[2].output,image_size/c.net_out_size[2],self.mini_batch_size,image_size)
        self.upscaled_output=T.concatenate([output1,output2,output3],axis=1)
        self.softmax.set_inpt(self.upscaled_output)
        self.output=self.softmax.output
        
    def cost(self):
        return -T.mean(T.dot(T.log(self.output),self.y))

    def accuracy(self):
        self.y_out = T.argmax(self.output, axis=1)
        return T.mean(T.eq(self.y, self.y_out))
    
    def SGD(self, epochs, eta, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        num_training_batches=c.num_training_batches
        num_validation_batches=c.num_validation_batches
        mini_batch_size=c.mini_batch_size
        train_images = open('train_images.pkl', 'rb')
        validate_images = open('validate_images.pkl', 'rb')
        train_labels = open('train_labels.pkl', 'rb')
        validate_labels = open('validate_labels.pkl', 'rb')
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(self.nets[0].layers[0].w1**2).sum() ,(self.nets[0].layers[0].w2**2).sum() ,(self.softmax.w**2).sum() ])
        cost = self.cost()+\
               0.5*lmbda*l2_norm_squared/c.train_size
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
        training_x=[]
        training_y=[]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        #theano.function(param,ouputs,updates,givens)
        #param-list of variables not shared variables
        #outputs-expressions to compute
        #updates-expressions for new shared variable values
        #givens-iterable over pairs (V1,V2) v1,v2 in a pair should have the same type

        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.input:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [], self.accuracy(),
            givens={
                self.input:
                validation_x,
                self.y:
                validation_y
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                temp=pickle.load(train_images)
                train_scale_1=shared(temp)
                temp=pickle.load(train_images)
                train_scale_2=shared(temp)
                temp=pickle.load(train_images)
                train_scale_3=shared(temp)
                temp=pickle.load(train_labels)
                y=shared(temp)
                training_x=[[train_scale_1,train_scale_2,train_scale_3]]
                training_y=[[y]]
                l=train_scale_3.shape.eval()
                for i in xrange(l[0]):
                    cost_ij = train_mb(minibatch_index*50+i)
                validation_accuracies=[]
                for j in xrange(num_validation_batches):
                    temp=pickle.load(validate_images)
                    validate_scale_1=shared(temp)
                    temp=pickle.load(validate_images)
                    validate_scale_2=shared(temp)
                    temp=pickle.load(validate_images)
                    validate_scale_3=shared(temp)
                    temp=pickle.load(validate_labels)
                    y=shared(temp)
                    validation_x=[[validate_scale_1,validate_scale_2,validate_scale_3]]
                    validation_y=[[y]]
                    validation_accuracies.append(validate_mb_accuracy())
                validation_accuracy = np.mean(validation_accuracies)
                    
        
        
