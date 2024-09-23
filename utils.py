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

def load_img(path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,img0=cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
    img0=cv2.resize(img0,dsize=(100,300))/255
    return torch.from_numpy(img0)

def convertXY2LR(Exy):
    Sq2=1/np.sqrt(2)
    conMat=torch.complex(torch.tensor([[1,0],[1,0]],dtype=torch.float32),torch.tensor([[0,-1],[0,1]],dtype=torch.float32))*Sq2
    if Exy.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Exy.shape
    Exy=torch.reshape(Exy,shape=(2,-1))
    Elr=torch.matmul(conMat,Exy)
    Elr=torch.reshape(Elr,shape=(2,width,height))
    return Elr

def convertAB2LR(Eab):
    conMat=torch.complex(torch.tensor([[1,1],[1,1]],dtype=torch.float32),torch.tensor([[-1,1],[1,-1]],dtype=torch.float32))*0.5
    if Eab.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Eab.shape
    Eab=torch.reshape(Eab,shape=(2,-1))
    Elr=torch.matmul(conMat,Eab)
    Elr=torch.reshape(Elr,shape=(2,width,height))
    return Elr

def convertLR2XY(Elr):
    Sq2=1/np.sqrt(2)
    conMat=torch.complex(torch.tensor([[1,1],[0,0]],dtype=torch.float32),torch.tensor([[0,0],[1,-1]],dtype=torch.float32))*Sq2
    if Elr.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Elr.shape
    Elr=torch.reshape(Elr,shape=(2,-1))
    Exy=torch.matmul(conMat,Elr)
    Exy=torch.reshape(Exy,shape=(2,width,height))
    return Exy

def convertLR2AB(Elr):
    conMat=torch.complex(torch.tensor([[1,1],[1,1]],dtype=torch.float32),torch.tensor([[1,-1],[-1,1]],dtype=torch.float32))*0.5
    if Elr.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Elr.shape
    Elr=torch.reshape(Elr,shape=(2,-1))
    Eab=torch.matmul(conMat,Elr)
    Eab=torch.reshape(Eab,shape=(2,width,height))
    return Eab

def complex2Afai(E0):
    amp1=torch.abs(E0[0,:,:]);amp2=torch.abs(E0[1,:,:])
    amp=torch.stack((amp1,amp2),0)
    fai=torch.angle(E0[0,:,:])-torch.angle(E0[1,:,:])   #左旋-右旋
    fai=(fai+2*torch.pi)%(2*torch.pi)
    return (amp,fai)

def convertLR2stocks(E0,img_size):
    conMat=torch.complex(torch.tensor([[1,1],[0,0]],dtype=torch.float32),torch.tensor([[0,0],[1,-1]],dtype=torch.float32))
    if E0.device !=conMat.device:
        conMat=conMat.cuda()
    Erl=torch.reshape(E0,shape=(2,-1))
    Exy=torch.matmul(conMat,Erl)
    Exy=torch.reshape(Exy,shape=(2,img_size,img_size))

    Exy_abs=abs(Exy); Exy_energy=Exy_abs*Exy_abs
    summ=torch.sum(Exy_energy,dim=(0,1,2)) #归一化
    Exy_energy=Exy_energy/summ

    S1=Exy_energy[0,:,:]-Exy_energy[1,:,:]
    EEE=Exy[0,:,:]*torch.conj(Exy[1,:,:])/summ
    S2=2*torch.real(EEE)
    S3=-2*torch.imag(EEE)
    Stocks=torch.stack((S1,S2,S3),0)
    return Stocks

def calc_phaseMask(args,path1,path2):
    img1=cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img1=cv2.resize(img1,dsize=(args.img_size,args.img_size))/255
    img2=cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
    img2=cv2.resize(img2,dsize=(args.img_size,args.img_size))/255
    phase=np.logical_or(img1,img2)
    pm=torch.from_numpy(phase)
    return pm
