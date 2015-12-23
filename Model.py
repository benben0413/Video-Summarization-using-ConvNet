from MultiScale import MultiScale
import pickle

def train_loader():
    nn=MultiScale()
    epochs=10
    eta=0.001
    lmbda=0.00001
    nn.SGD(epochs,eta,lmbda);
    f=file('neural_net.model','wb')
    pickle.dump(nn,f);
    f.close()


def test_loader():
    f=file('neural_net.model','rb')
    nn=pickle.load(f);
    f.close()

train_loader()
    
    
