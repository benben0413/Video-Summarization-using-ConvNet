import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from copy import deepcopy
import time
import cv2
import pickle
import theano

def attention_func(segments,upsampled_features):
    out=[]
    mask_for_components=[]
    num_components=[]
    masked_components=[]
    width=[(0,0),(0,0)]
    for segments_fz in segments:
        comps=np.unique(segments_fz)
        num_components.append(len(comps))
        for val in comps:
            points=[]
            a,b=np.where(segments_fz==val)
            for r,c in zip(a,b):
                points.append([[r,c]])
            x,y,w,h=cv2.boundingRect(np.asarray(points))
            print (x,y,w,h)
            temp=deepcopy(segments_fz[x:x+w,y:y+h])
            features=deepcopy(upsampled_features[:,x:x+w,y:y+h])
            print features
            temp[temp!=val]=0
            temp[temp==val]=1
            masked=np.multiply(temp,features)
            print masked
            if(w%3==0):
                width[0]=(0,0)
            elif(w%3==1):
                width[0]=(1,1)
            else:
                width[0]=(0,1)
            if(h%3==0):
                width[1]=(0,0)
            elif(h%3==1):
                width[1]=(1,1)
            else:
                width[1]=(0,1)
            
            padded=pad(masked,width).eval()
            print padded
            object_descriptor=elastic_max_pool(padded).eval()
            print (object_descriptor)
            time.sleep(3)
            masked_components.append(object_descriptor)
            out.append([x,y,w,h])
            
def pad(x, width, val=0, batch_ndim=1):
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    if val:
        out = T.ones(output_shape) * val
    else:
        out = T.zeros(output_shape)
    return T.set_subtensor(out[tuple(indices)], x)

def elastic_max_pool(inp):
    num_feature_maps,w,h=inp.shape
    maxpool_shape=(w/3,h/3)
    return downsample.max_pool_2d(inp,maxpool_shape)
    
f=file('train_segments.pkl','rb')
segments=pickle.load(f)
upsampled_features=np.random.random_integers(1,5,(2,256,256))
attention_func(segments,upsampled_features)
f.close()
    
