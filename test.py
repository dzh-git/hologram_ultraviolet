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

    path1='./dataset/newleft.png';path2='./dataset/newright.png'
    AL=load_img(args,path1)
    AL=torch.from_numpy(AL).float() 
    AR=load_img(args,path2)
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
    input_images_Total=input_images_Total/torch.sum(input_images_Total)


    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    outputs_L=model(input_images_Total)
    (pre_A,pre_phi)=outputs_L

    phi_norm=1/(2*torch.pi*args.img_size**2)
    criterion = torch.nn.MSELoss(reduction='sum')
    lossA=criterion(pre_A,target_A).float()
    lossphi=criterion(pre_phi,delta_phi).float()*phi_norm
    print("output1:",torch.sum(pre_A[0,:,:]),torch.max(pre_A[0,:,:]),torch.min(pre_A[0,:,:]))
    print("output2:",torch.sum(pre_A[1,:,:]),torch.max(pre_A[1,:,:]),torch.min(pre_A[1,:,:]))
    print('A:{:.9f},phi:{:.9f}'.format(lossA,lossphi))
    
    pre_A=pre_A.detach().numpy()
    pre_phi=pre_phi.detach().numpy()    

    plt.subplot(3,2,1)
    plt.imshow(AL,cmap='gray',vmin=0)
    plt.subplot(3,2,2)
    plt.imshow(pre_A[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,3)
    plt.imshow(AR,cmap='gray',vmin=0)
    plt.subplot(3,2,4)
    plt.imshow(pre_A[1,:,:],cmap='gray',vmin=np.min(pre_A[1,:,:]),vmax=np.max(pre_A[1,:,:]))
    plt.subplot(3,2,5)
    plt.imshow(delta_phi,cmap='gray',vmin=0)
    plt.subplot(3,2,6)
    plt.imshow(pre_phi,cmap='gray',vmin=0)

    plt.show()


if __name__=='__main__':
    main()

    
    