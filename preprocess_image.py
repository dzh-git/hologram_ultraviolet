import os
from functools import lru_cache
import numpy as np
import torch
import parameters
import random
import onn
from EarlyStop import EarlyStop
import cv2
import matplotlib.pyplot as plt
import utils

def load_img(path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,img0=cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
    img0=cv2.resize(img0,dsize=(100,300))/255
    return torch.from_numpy(img0)

def main():
    pathA='./dataset/a.png';pathB='./dataset/b.png'
    pathX='./dataset/x.png';pathY='./dataset/y.png'
    pathL='./dataset/l.png';pathR='./dataset/r.png'
    imgA=load_img(pathA);imgB=load_img(pathB)
    imgX=load_img(pathX);imgY=load_img(pathY)
    imgL=load_img(pathL);imgR=load_img(pathR)
    Eab=torch.stack((imgA,imgB),0).float();Eab=Eab/torch.sum(Eab)
    Exy=torch.stack((imgX,imgY),0).float();Exy=Exy/torch.sum(Exy)
    Elr=torch.stack((imgL,imgR),0).float();Elr=Elr/torch.sum(Elr)

    final_ab=torch.zeros([2,300,300]);    final_ab[:,:,0:100]=Eab
    final_xy=torch.zeros([2,300,300]);    final_xy[:,:,100:200]=Exy
    final_lr=torch.zeros([2,300,300]);    final_lr[:,:,200:300]=Elr
    torch.save(final_ab,'./dataset/Eab.pt')
    torch.save(final_xy,'./dataset/Exy.pt')
    torch.save(final_lr,'./dataset/Elr.pt')
    return 

    [width,height]=imgA.shape
    Eab=torch.complex(Eab,torch.zeros([2,width,height]).float())
    Exy=torch.complex(Exy,torch.zeros([2,width,height]).float())
    Elr=torch.complex(Elr,torch.zeros([2,width,height]).float())
    
    AB2LR=utils.convertAB2LR(Eab)
    XY2LR=utils.convertXY2LR(Exy)
    amp1,phi1=utils.complex2Afai(Elr)
    amp2,phi2=utils.complex2Afai(XY2LR)
    amp3,phi3=utils.complex2Afai(AB2LR)
    
    final_A=torch.zeros([2,300,300])
    final_A[:,:,0:100]=amp1;final_A[:,:,100:200]=amp2;final_A[:,:,200:]=amp3
    final_P=torch.zeros([300,300])
    final_P[:,0:100]=phi1;final_P[:,100:200]=phi2;final_P[:,200:]=phi3

    # plt.figure()
    # plt.imshow(final_A[0,:,:],cmap='gray')
    # plt.figure()
    # plt.imshow(final_A[1,:,:],cmap='gray')
    # plt.figure()
    # plt.imshow(final_P,cmap='gray')
    # plt.show()
    # gd_mask=final_A[0,:,:]+final_A[1,:,:]
    # gd_mask=gd_mask>0.1

    pre_Alr,_=utils.complex2Afai(pre_Elr)
    pre_Exy=utils.convertLR2XY(pre_Elr)
    pre_Axy,_=utils.complex2Afai(pre_Exy)
    pre_Eab=utils.convertLR2AB(pre_Elr)
    pre_Aab,_=utils.complex2Afai(pre_Eab)
    # torch.save(final_A,'./dataset/target_A.pt')
    # torch.save(final_P,'./dataset/target_P.pt')
    # torch.save(gd_mask,'./dataset/gd_mask.pt')


if __name__=='__main__':
    main()