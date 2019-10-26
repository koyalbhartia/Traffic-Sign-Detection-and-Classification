import argparse
import sys
import os, sys
import numpy as np
from numpy import linalg as LA
from numpy import linalg as la
from matplotlib import pyplot as plt
import math
from PIL import Image
import scipy.ndimage as nd
import random
from scipy.interpolate import RectBivariateSpline
import glob
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def Single_class(label,value):
    for i in range(len(label)):
        if label[i] not in value:
            label[i]=0
    return label
    

def deskew(img):
    img_red=img[:,:,2]
    SZ=64
    m = cv2.moments(img_red)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def HOG_descriptor(im):

    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    descriptor = hog.compute(im)
    descriptor=descriptor.T
    return descriptor

def SVM(C,gamma,train_Data,test_data,trainLabels):
    # incerase C better classification less C better margine
    NU=C
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_NU_SVC )
    svm.setKernel(cv2.ml.SVM_RBF)
    #12.5
    svm.setC(C)
    svm.setNu(NU)
    
    #0.50625
    svm.setGamma(gamma)
    svm.trainAuto(train_Data, cv2.ml.ROW_SAMPLE, trainLabels,10)
    svm.save("svm_model1.xml")
    # Test on a held out test set

    # testing
    testResponse = svm.predict(test_data)[1].ravel()
    print(testResponse)
    return testResponse

def resize(img):
    img=cv2.resize(img, (64,)*2).astype('uint8')
    return img

def getDescriptor_Mat(Path,desc_Mat,label_Mat,label):
    count=0
    for img in sorted(glob.glob(Path)):
        print(img)    
        print(count,"count")
        count+=1
        img=cv2.imread(img)
        img=resize(img)
        #cv2.imshow('img',img)
        #img=deskew(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img2',img)
        #cv2.waitKey(100)
        hog_desc=HOG_descriptor(img)
        #desc_Mat.append(hog_desc)
        #label_Mat.append(label)
        label_Mat=np.vstack((label_Mat,[label]))
        desc_Mat=np.vstack((desc_Mat,hog_desc))
    return desc_Mat,label_Mat

def descLabelMat():
    labels=[1,14,17,19,21,35,38,45,62,63]
    true_labels=[1,14,17,19,21,35,38,45]
    #labels=[1,45]
    #62,63
    traindesc_Mat=np.zeros((1,49*36),dtype='float32')
    testdesc_Mat=np.zeros((1,49*36),dtype='float32')
    #train_labels=np.zeros((1,8))
    train_labels=[0]
    test_labels=[0]

    for i in labels:
        print(i)
        if i in range(0,10): i='0'+str(i)
        else: i=str(i)
        if i=='62':
            TrainPath='Training/000'+i+'/*.png'
            TestPath='Testing/000'+i+'/*.png'
        else:   
            TrainPath='Training/000'+i+'/*.ppm'
            TestPath='Testing/000'+i+'/*.ppm'
        i=int(i)
        traindesc_Mat,train_labels=getDescriptor_Mat(TrainPath,traindesc_Mat,train_labels,i)
        testdesc_Mat,test_labels=getDescriptor_Mat(TestPath,testdesc_Mat,test_labels,i)
    testdesc_Mat=np.delete(testdesc_Mat,(0),axis=0)
    traindesc_Mat=np.delete(traindesc_Mat,(0),axis=0)
    test_labels=np.delete(test_labels,(0),axis=0)
    train_labels=np.delete(train_labels,(0),axis=0)
    #for i in range(0,63):      
    train_labels=Single_class(train_labels,true_labels)
    test_labels=Single_class(test_labels,true_labels)
    return traindesc_Mat,train_labels,testdesc_Mat,test_labels

def E_out(Prediction,test_labels):
    measure=0
    for i in range(len(Prediction)):
        if Prediction[i]==test_labels[i,0]:
           measure+=1 
    measure=measure/len(Prediction)
    return measure


traindesc_Mat,train_labels,testdesc_Mat,test_labels=descLabelMat()

print(traindesc_Mat)
print(testdesc_Mat)
print(train_labels)
print(test_labels)

answer=SVM(0.5,2,traindesc_Mat,testdesc_Mat,train_labels)

right=E_out(answer,test_labels)

print(right)



