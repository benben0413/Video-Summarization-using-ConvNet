import cv2
import pickle
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
datasetpath='/home/jyoti/MTP/Images/'
imagefiles =[ f for f in listdir(datasetpath) if isfile(join(datasetpath,f)) ]
train = open('train_images.pkl', 'wb')
validate = open('validate_images.pkl', 'wb')
scale_1_dataset=[]
scale_2_dataset=[]
scale_3_dataset=[]
count=0
num_files=0
""" Read imagefiles"""

for img in imagefiles:
    A = cv2.imread(join(datasetpath,img))
    if(A is not None):
        num_files=num_files+1
        """Separate R,G,B streams"""
        B=A[:,:,0]
        G=A[:,:,1]
        R=A[:,:,2]

        """initialise"""
        Y=np.ndarray((256,256),float)
        U=np.ndarray((256,256),float)
        V=np.ndarray((256,256),float)
        """Changing to YUV format"""
        Y_raw = 0.299*R + 0.587*G + 0.114*B
        U_raw = 0.492 *(B-Y_raw)
        V_raw = 0.877 *(R-Y_raw)
        cv2.GaussianBlur(Y_raw,(15,15),1,Y,1)
        cv2.GaussianBlur(U_raw,(15,15),1,U,1)
        cv2.GaussianBlur(V_raw,(15,15),1,V,1)

        """ generate Gaussian pyramid for Y channel """
        G = Y.copy()
        gpA = [G]
        for i in xrange(3):
            G = cv2.pyrDown(G)
            gpA.append(G)

        lpA_Y = [gpA[2]]
        for i in xrange(2,0,-1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtract(gpA[i-1],GE)
            lpA_Y.append(L)

        """ generate Gaussian pyramid for U channel """
        G = U.copy()
        gpA = [G]
        for i in xrange(3):
            G = cv2.pyrDown(G)
            gpA.append(G)

        lpA_U = [gpA[2]]
        for i in xrange(2,0,-1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtract(gpA[i-1],GE)
            lpA_U.append(L)

        """ generate Gaussian pyramid for V channel """
        G = V.copy()
        gpA = [G]
        for i in xrange(3):
            G = cv2.pyrDown(G)
            gpA.append(G)

        lpA_V = [gpA[2]]
        for i in xrange(2,0,-1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtract(gpA[i-1],GE)
            lpA_V.append(L)

        x,y=lpA_V[0].shape
        C=np.ndarray((3,x,y),float)
        C[0,:,:]=lpA_Y[0]
        C[1,:,:]=lpA_U[0]
        C[2,:,:]=lpA_V[0]
        scale_1_dataset.append(C)

        x,y=lpA_V[1].shape
        D=np.ndarray((3,x,y),float)
        D[0,:,:]=lpA_Y[1]
        D[1,:,:]=lpA_U[1]
        D[2,:,:]=lpA_V[1]
        scale_2_dataset.append(D)

        x,y=lpA_V[2].shape
        E=np.ndarray((3,x,y),float)
        E[0,:,:]=lpA_Y[2]
        E[1,:,:]=lpA_U[2]
        E[2,:,:]=lpA_V[2]
        scale_3_dataset.append(E)

        count=count+1
        if(num_files<=1800 and count == 50):
            pickle.dump(scale_1_dataset,train)
            pickle.dump(scale_2_dataset,train)
            pickle.dump(scale_3_dataset,train)
            scale_1_dataset=[]
            scale_2_dataset=[]
            scale_3_dataset=[]
            count=0
            print "In train"

        elif(count == 50):
            pickle.dump(scale_1_dataset,validate)
            pickle.dump(scale_2_dataset,validate)
            pickle.dump(scale_3_dataset,validate)
            scale_1_dataset=[]
            scale_2_dataset=[]
            scale_3_dataset=[]
            count=0
            print "In validate"
    

if(count!=0):
    pickle.dump(scale_1_dataset,validate)
    pickle.dump(scale_2_dataset,validate)
    pickle.dump(scale_3_dataset,validate)
    print "In validate"
validate.close()
train.close()
