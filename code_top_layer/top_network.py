import cPickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor import tanh

def shared(data):
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.ftensor4("x")
        self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, self.mini_batch_size)
        self.output = (self.layers[-1].output,self.layers[-1].y_out)

    def SGD(self, epochs, mini_batch_size, eta,lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        training_y=theano.shared(np.ndarray((1,1),dtype=theano.config.floatX),borrow=True)
        validation_x=theano.shared(np.ndarray((1,1,1,1),dtype=theano.config.floatX),borrow=True)
        validation_y=theano.shared(np.ndarray((1,1),dtype=theano.config.floatX),borrow=True)

        # compute number of minibatches for training, validation and testing
        num_training_batches = 360
        num_validation_batches = 180
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].divergence(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        
        for epoch in xrange(epochs):
            f1=file('train_top_layer.pkl','rb')
            f2=file('validate_top_layer.pkl','rb')
            validation_accuracies=[]
            for minibatch_index in xrange(num_training_batches):
                x,y=pickle.load(train_images)
                training_x.set_value(x)
                training_y.set_value(y)
                l=training_x.shape.eval()
                for i in xrange(l[0]):
                    cost_ij = train_mb(i)
                    print cost_ij
            for minibatch_index in xrange(num_validation_batches):
                x,y=pickle.load(validation_images)
                validation_x.set_value(x)
                validation_y.set_value(y)
                l=validation_x.shape.eval()
                for i in xrange(l[0]):
                    validation_accuracies.append(validate_mb_accuracy(i))
             validation_accuracy = np.mean(validation_accuracies)
             print validation_accuracy


#### Define layer types

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=tanh):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        # Initialize weights 
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.params = [self.w]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        "predicted label distribution for the component "
        self.output = softmax(T.dot(self.inpt, self.w))
        "predicted label for the component "
        self.y_out = T.argmax(self.output, axis=1)

    def prediction(self):
        "index of nonzero elements in output"
        temp=T.nonzero(self.output)[0]
        "selected nonzero elements of output"
        output=T.take(self.output,temp)
        "predicted confidence cost, predicted label "
        return -T.sum(output * T.log10(output)),self.y_out

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(T.argmax(y, axis=1), self.y_out))

    def divergence(self, net):
        "Return the KL divergence."
        "index of nonzero elements in output"
        temp=T.nonzero(self.output)[0]
        "remove all zero elements of predicted output and ground truth to avoid division by zero and log of zero"
        y1=T.take(net.y,temp)
        output1=T.take(self.output,temp)
        temp=T.nonzero(y1)[0]
        y=T.take(y1,temp)
        output=T.take(output1,temp)
        return T.sum(y * T.log10(output/y))




