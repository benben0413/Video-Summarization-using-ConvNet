import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from collections import Counter
from copy import deepcopy
import cv2,sys
import pickle
import random
import theano

def attention_func(segments,upsampled_features,labels):
    num_components=[]
    masked_components=[]
    true_distribution=[]
    width=[(0,0),(0,0)]
    for segments_fz in segments:
        comps=np.unique(segments_fz)
        num_components.append(len(comps))
        for val in comps:
            points=[]
            temp=[]
            a,b=np.where(segments_fz==val)
            for r,c in zip(a,b):
                points.append([[r,c]])
                temp.append(labels[r,c])
            k=Counter(temp)
            temp=[]
            for a in range(33):
                temp.append(k[a])
            s=sum(temp)
            for a in range(33):
                temp[a]=float(temp[a])/s
            true_distribution.append(temp)
            x,y,w,h=cv2.boundingRect(np.asarray(points))
            temp=deepcopy(segments_fz[x:x+w,y:y+h])
            features=deepcopy(upsampled_features[:,x:x+w,y:y+h])
            temp[temp!=val]=0
            temp[temp==val]=1
            masked=np.multiply(temp,features)
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
            object_descriptor=elastic_max_pool(padded).eval()
            masked_components.append(object_descriptor)
    return masked_components,true_distribution
            
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

args=sys.argv
if(len(args) < 5):
    print "Command line args missing : args[1] - segments file , args[2] - scale invariant features file , args[3] - labels file, args[4]- output file"
    sys.exit(-1)
f1=file(args[1],'rb')
f2=file(args[2],'rb')
f3=file(args[3],'rb')
f4=file(args[4],'wb')
"""f1=file('train_segments.pkl','rb')
f2=file('train_scale_invariant_features.pkl','rb')
f3=file('train_labels.pkl','rb')
f4=file('train_top_layer.pkl','wb')"""
for i in range(36):
    label_list=pickle.load(f3)
    descriptors=[]
    ground_distribution=[]
    for k,labels in enumerate(label_list):
        segments=pickle.load(f1)
        temp1,temp2=attention_func(segments,upsampled_features,labels)
        descriptors=descriptors+temp1
        ground_distribution=ground_distribution+temp2
        if(k!=0 and (k+1)%5==0):
            random.shuffle(descriptors)
            random.shuffle(ground_distribution)
            pickle.dump((descriptors,ground_distribution),f4)
            descriptors=[]
            ground_distribution=[]
f1.close()
f2.close()
f3.close()
f4.close()
    
