#### Libraries
# Standard library
import pickle
import cPickle
# Third-party libraries
import numpy as np
import Constants as c
import theano
from network import Network,ConvPoolLayer,RandCombConvLayer1, RandCombConvLayer2
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
import functions,random
from sklearn import preprocessing


# Activation functions for neurons
from theano.tensor import tanh

GPU = False
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
        self.mini_batch_size = c.mini_batch_size
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt):
        self.inpt = inpt.dimshuffle(0,2,3,1)
	o=T.dot(self.inpt,self.w)+self.b
        self.output = softmax(o.reshape((256*256,34)))
        self.y_out = T.argmax(self.output, axis=1)
        

class MultiScale(object):

    def __init__(self):
        """Takes a list of `networks`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        filter_shape1=c.filter_shape1
        filter_shape2=c.filter_shape2
        filter_shape3=c.filter_shape3
        filter_shape4=c.filter_shape4
        poolsize=c.poolsize
        n_out1=(np.prod(filter_shape1)/np.prod(poolsize))
        n_out2=(np.prod(filter_shape2)/np.prod(poolsize))
        n_out3=(np.prod(filter_shape3)/np.prod(poolsize))
        n_out4=(np.prod(filter_shape4)/np.prod(poolsize))
        self.wt64=[]
        self.wt256=[]
        rand_comb=[]
        rand_comb2=[]
        self.wt_conv=[theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out1),size=filter_shape1),dtype=theano.config.floatX),borrow=True),
                      theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out2),size=filter_shape2),dtype=theano.config.floatX),borrow=True)]
        self.b_conv= theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(c.feature_map_count[0])),dtype=theano.config.floatX),borrow=True)
        for i in range(64):
            rand_selection=random.sample([k for k in range(16)],8)
            rand_comb.append(rand_selection)
            self.wt64.append(theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out3),size=filter_shape3),dtype=theano.config.floatX),borrow=True))
        self.b64= theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(c.feature_map_count[1])),dtype=theano.config.floatX),borrow=True)
        for i in range(256):
            rand_selection=random.sample([k for k in range(64)],32)
            rand_comb2.append(rand_selection)
            self.wt256.append(theano.shared(np.asarray(np.random.normal(loc=0,scale=np.sqrt(1.0/n_out4),size=filter_shape4),dtype=theano.config.floatX),borrow=True))

        self.input1=T.tensor4("input1")
        self.input2=T.tensor4("input2")
        self.input3=T.tensor4("input3")
        self.y=T.tensor4("y")
        self.nets=[Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombConvLayer1(self.wt64,self.b64,rand_comb),
            RandCombConvLayer2(self.wt256,rand_comb2)],self.input1,c.image_shape1),
                   Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombConvLayer1(self.wt64,self.b64,rand_comb),
            RandCombConvLayer2(self.wt256,rand_comb2)],self.input2,c.image_shape2),
                   Network([
            ConvPoolLayer(self.wt_conv,self.b_conv),
            RandCombConvLayer1(self.wt64,self.b64,rand_comb),
            RandCombConvLayer2(self.wt256,rand_comb2)],self.input3,c.image_shape3)]

        self.num_class=c.num_class
        self.softmax=SoftmaxLayer(768,self.num_class)
        self.mini_batch_size = c.mini_batch_size
        self.params =[w for w in self.wt_conv]+self.wt64+[self.b64]+self.wt256+ [self.b_conv]+ self.softmax.params
        image_size=c.image_size
        output1=functions.upsample(self.nets[0].output,image_size/c.net_out_size[0])
        output2=functions.upsample(self.nets[1].output,image_size/c.net_out_size[1])
        output3=functions.upsample(self.nets[2].output,image_size/c.net_out_size[2])
        self.upscaled_output=T.concatenate([output1,output2,output3],axis=1)
        self.softmax.set_inpt(self.upscaled_output)
        self.output=self.softmax.output
        del filter_shape1,filter_shape2,filter_shape3,filter_shape4,poolsize,n_out1,n_out2,n_out3,n_out4,output1,output2,output3,image_size,rand_comb,rand_comb2,rand_selection
        
    def cost(self):
       	return -T.mean(T.log(self.output.reshape((1,256,256,34)))*self.y)

    def accuracy(self):
        self.y_out = T.argmax(self.output, axis=1)
	y=T.argmax(self.y.reshape((256*256,34)),axis=1)
        return T.mean(T.eq(y, self.y_out))
    
    def SGD(self,epochs,eta, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(self.nets[0].layers[0].w1**2).sum() ,(self.nets[0].layers[0].w2**2).sum(),(sum([w**2 for w in self.nets[0].layers[1].wt])).sum()\
                               ,(sum([w**2 for w in self.nets[0].layers[2].wt])).sum(),(self.softmax.w**2).sum() ])
        cost = self.cost()+ 0.5*lmbda*l2_norm_squared/c.train_size
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) for param,grad in zip(self.params, grads)]
        train1=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        train2=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        train3=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        training_y=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        validation1=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        validation2=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        validation3=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        validation_y=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        
        i = T.lscalar() # mini-batch index

        print "SGD"

        train_mb = theano.function(
            [i],cost,updates=updates,
            givens={
                self.input1:
                train1[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.input2:
                train2[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.input3:
                train3[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y                
            })
        print "train_mb"
        validate_mb_accuracy = theano.function(
            [i], self.accuracy(),
            givens={
                self.input1:
                validation1[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.input2:
                validation2[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.input3:
                validation3[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y                
            })
        print "phew"

        lb = preprocessing.LabelBinarizer()
        binary_encoded=np.ndarray((1,256,256,34))
        lb.fit([k for k in xrange(34)])              
        # Do the actual training
        for epoch in xrange(epochs):
            train_images = open('../train_images.pkl', 'rb')
            validate_images = open('../validate_images.pkl', 'rb')
            train_labels = open('../train_labels.pkl', 'rb')
            validate_labels = open('../validate_labels.pkl', 'rb')
            for k in xrange(c.num_training_batches):
                train1.set_value(pickle.load(train_images))
                train2.set_value(pickle.load(train_images))
                train3.set_value(pickle.load(train_images))
                temp=pickle.load(train_labels)
                l=train1.shape.eval()               
                for i in xrange(l[0]):
                    for v in xrange(256):
                    	binary_encoded[0,v,:,:]=lb.transform(temp[i][v])
                    training_y.set_value(binary_encoded)
                    cost_ij = train_mb(i)
                    print i,cost_ij
            validation_accuracies=[]
            for j in xrange(c.num_validation_batches):
                validation1.set_value(pickle.load(validate_images))
                validation2.set_value(pickle.load(validate_images))
                validation3.set_value(pickle.load(validate_images))
                temp=pickle.load(validate_labels)
	        l=validation1.shape.eval()
	        for i in xrange(l[0]):
                    for v in xrange(256):
                        binary_encoded[0,v,:,:]=lb.transform(temp[i][v])
                    validation_y.set_value(binary_encoded)
                    validation_accuracies.append(validate_mb_accuracy(i))
            validation_accuracy = np.mean(validation_accuracies)
            print validation_accuracy
            train_images.close()
            validate_images.close()
            train_labels.close()
            validate_labels.close()

        f=file('neural_net_weights.pkl','wb')
        for param in self.params:
            print param.eval()
            cPickle.dump(param.eval(),f);
        f.close()
        
        
        
