import cv2
import pickle
import scipy.io
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
ifile=scipy.io.loadmat('/home/jyoti/MTP/annotationIndexdData.mat')
labels=ifile['data']
train = open('train_labels.pkl', 'wb')
validate = open('validate_labels.pkl', 'wb')
count=0
label_list=[]
num_files=0
""" Read imagefiles"""

for label in labels:
        num_files=num_files+1
        count=count+1
        label_list.append(label)
        if(num_files<=1800 and count == 50):
            pickle.dump(label_list,train)
            label_list=[]
            count=0
            print "In train"

        elif(count == 50):
            pickle.dump(label_list,validate)
            label_list=[]
            count=0
            print "In validate"
    

if(count!=0):
        pickle.dump(label_list,validate)
        print "In validate"
validate.close()
train.close()
