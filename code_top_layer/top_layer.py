import cPickle
import numpy as np
import cv2
import theano
from top_network import Network,FullyConnectedLayer,SoftmaxLayer
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
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

net=Network([FullyConnectedLayer(6912,1024,activation_fn=tanh),SoftmaxLayer(1024,33)],1)

f1=file('top_layer_model.pkl','wb')
epochs=20
mini_batch_size=1
eta=0.01
"""for i in range(200):
    descriptors,ground_distribution=cPickle.load(f1)
    training_data=(shared(descriptors),shared(ground_distribution))
    desc,distribution=cPickle.load(f2)
    validation_data=(shared(desc),shared(distribution))"""
net.SGD(epochs, mini_batch_size,eta,0.01)
cPickle.dump(net,f1)
f1.close()
