import os
from functools import lru_cache
import numpy as np
import torch
import parameters
import random
import onn
import csv
from EarlyStop import EarlyStop
import cv2
import matplotlib.pyplot as plt
import utils

def load_img(args,path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img0=cv2.resize(img0,dsize=(args.img_size,args.img_size))
    #归一化，总能量为1
    img0=img0/np.sum(img0)
    return img0

def main():
    args=parameters.my_parameters().get_hyperparameter()

    AL=load_img(args,'./dataset/newleft.png')
    AL=torch.from_numpy(AL).float() 
    AR=load_img(args,'./dataset/newright.png')
    AR=torch.from_numpy(AR).float() 
    target_A=torch.stack((AL,AR),0)
    delta_phi=torch.zeros((args.img_size,args.img_size))
    delta_phi[args.img_size//2:,:args.img_size//2]=torch.pi/2 ; delta_phi[args.img_size//2:,args.img_size//2:]=torch.pi
    delta_phi[:args.img_size//2,args.img_size//2:]=torch.pi*3/2
    #入射光
    pixel_num=args.img_size*args.img_size
    input_images=torch.ones(size=[2,args.img_size,args.img_size])/pixel_num 
    input_images_Total=torch.stack([torch.ones([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_R=torch.stack([torch.zeros([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()

    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.eval()
    pre_Elr=model(input_images_Total)
    pre_Alr,_=utils.complex2Afai(pre_Elr)
    pre_Exy=utils.convertLR2XY(pre_Elr)
    pre_Axy,_=utils.complex2Afai(pre_Exy)
    pre_Eab=utils.convertLR2AB(pre_Elr)
    pre_Aab,_=utils.complex2Afai(pre_Eab)
    

    pre_Alr=pre_Alr.detach().numpy()
    pre_Axy=pre_Axy.detach().numpy()
    pre_Aab=pre_Aab.detach().numpy()

    plt.subplot(3,2,1)
    plt.imshow(pre_Aab[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,2)
    plt.imshow(pre_Aab[1,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,3)
    plt.imshow(pre_Axy[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,4)
    plt.imshow(pre_Axy[1,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,5)
    plt.imshow(pre_Alr[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,6)
    plt.imshow(pre_Alr[1,:,:],cmap='gray',vmin=0)

    plt.savefig('result/vectorial_hologram_1.png')
    plt.show()

if __name__=='__main__':
    main()

    
    