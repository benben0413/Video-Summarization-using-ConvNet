from MultiScale import MultiScale
import pickle
import theano
import Constants as c
import numpy as np
import sys
sys.setrecursionlimit(50000)

def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

def train(nn,epochs,eta,lmbda):
    """Train the network using mini-batch stochastic gradient descent."""
    num_training_batches=c.num_training_batches
    num_validation_batches=c.num_validation_batches
    mini_batch_size=c.mini_batch_size
    train_images = open('train_images.pkl', 'rb')
    validate_images = open('validate_images.pkl', 'rb')
    train_labels = open('train_labels.pkl', 'rb')
    validate_labels = open('validate_labels.pkl', 'rb')
    temp=[]
    temp1=[]
    temp2=[]
    temp3=[]
    # Do the actual training
    for j in xrange(num_validation_batches):
        temp1+=pickle.load(validate_images)
        temp2+=pickle.load(validate_images)
        temp3+=pickle.load(validate_images)
        temp+=pickle.load(validate_labels)
    validate_scale_1=shared(temp)
    validate_scale_2=shared(temp)
    validate_scale_3=shared(temp)
    y=shared(temp)
    validation_data=(validate_scale_1,validate_scale_2,validate_scale_3,y)
            
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
            training_y=y
            training_data=(train_scale_1,train_scale_2,train_scale_3,training_y)
            nn.SGD(training_data,validation_data,epochs,eta)
   
def train_loader():
    nn=MultiScale()
    epochs=10
    eta=0.001
    lmbda=0.00001
    train(nn,epochs,eta,lmbda);
    f=file('neural_net_model.pkl','wb')
    pickle.dump(nn,f);
    f.close()


def test_loader():
    f=file('neural_net_model.pkl','rb')
    nn=pickle.load(f);
    f.close()

train_loader()
    
    
