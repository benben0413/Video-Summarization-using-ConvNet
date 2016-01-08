import pickle

import numpy as np
import Constants as c 
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
import random 
import functions 
from theano.tensor import tanh



# Later to remove image shape as input to the network init

class Network(object):

    def __init__(self,layers,x,image_shape):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.image_shape=image_shape
        self.x = x
        init_layer= self.layers[0]
        init_layer.set_inpt(self.x,self.image_shape,c.mini_batch_size)
        f=c.random_num_filters
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output,self.image_shape,c.mini_batch_size)
        self.output = self.layers[-1].output

#### Define layer types

class ConvPoolLayer(object):
    """ Used to create a combination of a convolutional and a max-pooling
    layer. """

    def __init__(self,wt_conv,b_conv):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape1 =c.filter_shape1
        self.filter_shape2 =c.filter_shape2
        self.poolsize = c.poolsize
        self.activation_fn=c.activation_fn

        self.w1 = wt_conv[0]        
        self.w2 = wt_conv[1]
        self.b = b_conv

    def set_inpt(self,inpt,image_shape,mini_batch_size):
        self.inpt=inpt.reshape(image_shape)
        self.Y=self.inpt[:,0:1,:,:]
        self.UV=self.inpt[:,1:3,:,:]
        self.padding = c.padding
        self.YPadded = functions.pad(self.Y,self.padding)
        self.UVPadded = functions.pad(self.UV,self.padding)

        conv_out_Y = conv.conv2d(
            input=self.YPadded, filters=self.w1, filter_shape=self.filter_shape1)
        conv_out_UV = conv.conv2d(
            input=self.UVPadded, filters=self.w2, filter_shape=self.filter_shape2)
        # what is the dimension of output coming from here ?

        conv_out=tensor.concatenate([conv_out_Y,conv_out_UV],axis=1)
        activation=self.activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        pooled_out = downsample.max_pool_2d(
            input=activation, ds=self.poolsize, ignore_border=True)
        self.output=pooled_out
        

class RandCombConvLayer1(object):
    """ Used to create a combination of a convolutional and a max-pooling
    layer. """

    def __init__(self,wt,b,filter_shape):
        """
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        """
        self.activation_fn=c.activation_fn
        self.wt = wt        
        self.b = b
        self.filter_shape = c.filter_shape3
        self.padding = c.padding

    def set_inpt(self,inpt,image_shape,mini_batch_size):
        self.inpt=inpt
        self.inptPadded = functions.pad(self.inpt,self.padding)
        conv_in_random = random.sample(self.inptPadded,c.random_num_filters[0])
        for i in range(7):
            conv_in_random = tensor.concatenate([conv_in_random,random.sample(self.inptPadded),8])
        conv_out = conv.conv2d(
            input = conv_in_random, filters = self.wt, filter_shape = self.filter_shape)
        activation=self.activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        pooled_out = downsample.max_pool_2d(
            input=activation, ds=self.poolsize, ignore_border=True)
        self.output=pooled_out

        #for i in xrange(mini_batch_size):
        #    inp=inpt[i].T
        #    t=tensor.dot(inp,self.wt)
        #    k=t.T
        #    j=k.reshape((1,n,s[2],s[3]))
        #    out.append(t.T)
            
        #output=tensor.concatenate(out,axis=0)
        #activation=self.activation_fn(output + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output=activation

class RandCombConvLayer2(object):

    def __init__(self,wt,b):
        """
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        """
        self.activation_fn=c.activation_fn
        self.wt = wt        
        self.b = b
        self.filter_shape=c.filter_shape4
        self.padding = c.padding

    def set_inpt(self,inpt,image_shape,mini_batch_size):
        self.inpt=inpt
        self.inptPadded = functions.pad(self.inpt,self.padding)
        conv_in_random = random.sample(self.inptPadded,c.random_num_filters[1])
        for i in range(7):
            conv_in_random = tensor.concatenate([conv_in_random,random.sample(self.inptPadded),32])
        conv_out = conv.conv2d(
            input = conv_in_random, filters = self.wt, filter_shape = self.filter_shape)
        #activation=self.activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #pooled_out = downsample.max_pool_2d(
        #    input=activation, ds=self.poolsize, ignore_border=True)
        self.output=pooled_out



        #self.inpt=inpt
        #s=list(image_shape)
        #s[2]=(s[2]-6)/2
        #s[3]=(s[3]-6)/2
        #out=[]
        #for i in xrange(mini_batch_size):
        #    inp=inpt[i].T
        #    t=tensor.dot(inp,self.wt)
        #    k=t.T
        #    j=k.reshape((1,n,s[2],s[3]))
        #    out.append(t.T)
            
        #output=tensor.concatenate(out,axis=0)
        #activation=self.activation_fn(output + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output=activation

