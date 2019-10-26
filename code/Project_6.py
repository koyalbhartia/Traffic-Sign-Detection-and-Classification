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

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def denoising_image(contrast_eq_image):
    denoised_image = cv2.fastNlMeansDenoisingColored(contrast_eq_image,None,10,10,7,21)
    return denoised_image

def Contrast_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    contarst_eq_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return contarst_eq_image

def Convert_to_hsv(denoised_image):
    hsv_image=cv2.cvtColor(denoised_image,cv2.COLOR_BGR2HSV)
    return hsv_image

def threshold_red(hsv_image,image):
    Final=image*0
    lower_bound=np.array([0,40,30])
    upper_bound=np.array([10,255,255])
    red_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    Final[:,:,0]=cv2.bitwise_and(image[:,:,0],red_mask)
    Final[:,:,1]=cv2.bitwise_and(image[:,:,1],red_mask)
    Final[:,:,2]=cv2.bitwise_and(image[:,:,2],red_mask)
    #kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #Final=cv2.morphologyEx(Final,cv2.MORPH_CLOSE,kernel)
    return Final

def threshold_blue(hsv_image,image):
    Final=image*0
    lower_bound=np.array([100,50,50])
    upper_bound=np.array([110,255,255])
    #100,50,50
    #110,255,255
    blue_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    Final[:,:,0]=cv2.bitwise_and(image[:,:,0],blue_mask)
    Final[:,:,1]=cv2.bitwise_and(image[:,:,1],blue_mask)
    Final[:,:,2]=cv2.bitwise_and(image[:,:,2],blue_mask)

    return Final

def deskew(img):
    img_red=img[:,:,2]
    SZ=64
    m = cv2.moments(img_red)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def Imageprocessor(image):
    # Crop image
    imageROI = image[0:int(1236*0.5),:,:]
    black_down=np.zeros((int(1236*0.5),1628,3))
    contrast_eq_image=Contrast_equalization(imageROI)
    denoised_image=denoising_image(contrast_eq_image)
    black=np.vstack((denoised_image,black_down))
    black=black.astype(np.uint8)
    print(np.shape(black))
    return black

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


def falsepositives(img,regions,a1,a2):
    high_area=0
    p1=[(0,0),(1,1)]
    for p in regions[0]:
        area = cv2.contourArea(p.reshape(-1, 1, 2))
        if area>a1 and area<a2:
            x,y,w,h = cv2.boundingRect(p.reshape(-1, 1, 2))
            ratio=h/w
            if ratio<1.6 and ratio>1.3:
                #1.6,1.3
                cv2.rectangle(img,(np.abs(x),np.abs(y)),(x+h,y+h),(0,255,0),3)
                cv2.imshow('img', img)
                if high_area<area:
                    high_area=area
                    p1=[(np.abs(int(y/2)-10),np.abs(int(x/2)-10)),(np.abs(int(y/2+h/2)+10),np.abs(int(x/2+w/2)+10))]
                    print(p1)
            elif ratio<1.1 and ratio>0.9:
                cv2.rectangle(img,(np.abs(x),np.abs(y)),(x+h,y+h),(0,255,0),3)
                cv2.imshow('img', img)
                if high_area<area:
                    high_area=area
                    p1=[(np.abs(int(y/2)-10),np.abs(int(x/2)-10)),(np.abs(int(y/2+h/2)+10),np.abs(int(x/2+w/2)+10))]
                    print(p1)
    #cv2.imshow('img', img)
    return img,p1

def detection(image,a1,a2):
    mser = cv2.MSER_create()
    image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions = mser.detectRegions(gray)
    vis = image.copy()
    img,p1=falsepositives(vis,regions,a1,a2)
    # cv2.polylines(vis, hull, 1, (0,255,0))
    return p1


def crop(image,coord):
    crop = image[coord[0][0]:coord[1][0],coord[0][1]:coord[1][1]]
    cv2.imshow('crop',crop)
    crop=cv2.resize(crop, (64,64)).astype('uint8')
    return crop

def SVMpredict(crop,svm):
    #svm = cv2.ml.SVM_create()

    crop=deskew(crop)
    test_hog=HOG_descriptor(crop)
    #svm.load("svm_model.xml")
    predict=svm.predict(test_hog)[1].ravel()
    return predict

def attach(predict,image,coord):

    if int(predict)!=0:
        w=int(coord[1][1]-coord[0][1])
        h=int(coord[1][0]-coord[0][0])
        if int(w)>5 and int(h)>5 and coord[1][1]+w<1600 and coord[0][0]+h<1200:
            attach=cv2.imread('%d.ppm'%predict)
            print(predict)
            print(np.shape(attach))
            attach = cv2.resize(attach, (w,h))
            mean=(coord[1][1]-coord[0][1])/2
            if mean >= 814 and coord[0][1]>w:
                image[coord[0][0]:coord[0][0]+h,coord[0][1]-w:coord[0][1]]=attach
                image=cv2.rectangle(image, (coord[0][1],coord[0][0]), (coord[1][1],coord[1][0]), (0,255,0), 2)

            else:
                image[coord[0][0]:coord[0][0]+h,coord[1][1]:coord[1][1]+w]=attach
                image=cv2.rectangle(image, (coord[0][1],coord[0][0]), (coord[1][1],coord[1][0]), (0,255,0), 2)
    return image

def Pipeline():
    vidObj = cv2.VideoCapture()
    count=0
    img_array=[]
    svm = cv2.ml.SVM_load("svm_model.xml")
    for i in range(32720,32720+500):
        #35500
        image=cv2.imread('input/image.0%d.jpg'%i)
        orig_img=image.copy()
        height,width,layers=image.shape
        size = (width,height)
        #cv2.imshow('org_img',image)
        denoised_image=Imageprocessor(image)
        hsv_image=Convert_to_hsv(denoised_image)
        #cv2.imshow('hsv',hsv_image)
        red=hsv_image.copy()
        blue=hsv_image.copy()
        blue_mask=threshold_blue(blue,orig_img)
        red_mask=threshold_red(red,orig_img)
        #print(np.shape(blue_mask))
        cv2.imshow('red',red_mask)
        p1=detection(blue_mask,1700,4000)
        p2=detection(red_mask,1500,4000)
        ### red images
        crp_img_r=crop(orig_img,p2)
        cv2.imshow('crop red',crp_img_r)

        predict_r=SVMpredict(crp_img_r,svm)
        Final_r=attach(predict_r,orig_img,p2)
        ### blue images
        crp_img_b=crop(orig_img,p1)
        predict_b=SVMpredict(crp_img_b,svm)
        #print(predict_b,'predict baba')
        Final_b=attach(predict_b,Final_r,p1)

        cv2.imshow('Final',Final_b)
        cv2.waitKey(10)
        print(count)
        count += 1
        print('Frame processing index')
        print(i)
        #cv2.imwrite('%d.jpg' %count,Final_b)
        img_array.append(Final_b)
        #success, image = vidObj.read()
    return img_array,size

def video(img_array,size):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video=cv2.VideoWriter('video.avi',fourcc, 15.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()

# main
if __name__ == '__main__':

    # Calling the function
    Image,size=Pipeline()
    video(Image,size)
