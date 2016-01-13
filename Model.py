from MultiScale import MultiScale
import pickle
import theano
import Constants as c
import numpy as np
import sys
sys.setrecursionlimit(10000)

def train_loader():
    nn=MultiScale()
    epochs=10
    eta=0.001
    lmbda=0.00001
    nn.SGD(epochs,eta,lmbda);
    f=file('neural_net_model.pkl','wb')
    pickle.dump(nn,f);
    f.close()


def test_loader():
    f=file('neural_net_model.pkl','rb')
    nn=pickle.load(f);
    f.close()

train_loader()
    
    
