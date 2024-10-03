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
import scipy

def load_img(args,path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img0=cv2.resize(img0,dsize=(args.img_size,args.img_size))
    #归一化，总能量为1
    img0=img0/np.sum(img0)
    return img0

def show_result():
    args=parameters.my_parameters().get_hyperparameter()

    AL=torch.from_numpy(np.load('./dataset/AL.npy'))
    AR=torch.from_numpy(np.load('./dataset/AR.npy'))
    target_A=torch.stack((AL,AR),0)
    delta_phi=torch.from_numpy(np.load('./dataset/delta_phi.npy'))
    amplitude_mask=torch.from_numpy(np.load('./dataset/amplitude_mask.npy'))
    target_A=target_A*amplitude_mask
    delta_phi=delta_phi*amplitude_mask
    #入射光
    input_images_Total=torch.stack([torch.ones([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_R=torch.stack([torch.zeros([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_Total=input_images_Total/torch.sum(input_images_Total)*1e6


    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.eval()
    outputs_L=model(input_images_Total)
    (pre_A,pre_phi)=outputs_L
    
    show_A=pre_A.detach().numpy()
    show_phi=pre_phi.detach().numpy()    
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(show_A[0,:,:],cmap='gray',vmin=0)
    plt.subplot(1,2,2)
    plt.imshow(show_A[1,:,:],cmap='gray',vmin=0)

    plt.figure()
    plt.subplot(3,2,1)
    plt.imshow(AL,cmap='gray',vmin=0)
    plt.subplot(3,2,2)
    plt.imshow(show_A[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,3)
    plt.imshow(AR,cmap='gray',vmin=0)
    plt.subplot(3,2,4)
    plt.imshow(show_A[1,:,:],cmap='gray',vmin=np.min(show_A[1,:,:]),vmax=np.max(show_A[1,:,:]))
    plt.subplot(3,2,5)
    plt.imshow(delta_phi,cmap='gray',vmin=0,vmax=2*torch.pi)
    plt.subplot(3,2,6)
    plt.imshow(show_phi,cmap='gray',vmin=0,vmax=2*torch.pi)
    plt.show()

    
    phi_norm=1/(2*torch.pi*args.img_size**2)
    criterion = torch.nn.MSELoss(reduction='mean')
    pre_A=pre_A*amplitude_mask
    pre_phi=pre_phi*amplitude_mask
    lossA=criterion(pre_A,target_A).float()
    lossphi=criterion(pre_phi,delta_phi).float()*phi_norm
    print("output1:",torch.sum(pre_A[0,:,:]),torch.max(pre_A[0,:,:]),torch.min(pre_A[0,:,:]))
    print("output2:",torch.sum(pre_A[1,:,:]),torch.max(pre_A[1,:,:]),torch.min(pre_A[1,:,:]))
    print('A:{:.9f},phi:{:.9f}'.format(lossA,lossphi))

def show_output_np(phase):
    args=parameters.my_parameters().get_hyperparameter()
    actual_situation = parameters.my_parameters().get_actualparameter()
    distance=actual_situation.distance
    wave_length=actual_situation.wave_length
    screen_length=actual_situation.screen_length
    wave_num=2*3.14159/wave_length
    point_num=args.img_size; dx=screen_length/point_num

    fx_list=np.arange(-1/(2*dx),1/(2*dx),1/screen_length)
    phi = np.fromfunction(
        lambda i, j: 1-(np.square(wave_length*fx_list[i])+np.square(wave_length*fx_list[j])),
        shape=(point_num, point_num), dtype=np.int16).astype(np.complex64)
    H = np.exp(1.0j * wave_num * distance*np.sqrt(phi))
    H = np.fft.fftshift(H)

    left=np.exp(1.0j *phase)
    right=np.exp(-1.0j *phase)
    LEFT = np.fft.fft2(np.fft.fftshift(left))
    RIGHT = np.fft.fft2(np.fft.fftshift(right))

    k_space_L=LEFT*H
    k_space_R=RIGHT*H
    x_space_l = np.fft.ifftshift(np.fft.ifft2(k_space_L))
    x_space_r = np.fft.ifftshift(np.fft.ifft2(k_space_R))

    res_angle=np.angle(x_space_r)-np.angle(x_space_l)
    res_angle=res_angle%(2*torch.pi)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(abs(x_space_l),cmap='gray',vmin=0)
    plt.subplot(1,3,2)
    plt.imshow(abs(x_space_r),cmap='gray',vmin=0)
    plt.subplot(1,3,3)
    plt.imshow(res_angle,cmap='gray',vmin=0,vmax=np.pi*2)
    plt.show()

def quantize_tensor(x):
    scale=2*np.pi/8
    q_x=x/scale
    q_x=q_x.round()%8
    return q_x

def dequantize_tensor(q_x):
    scale=2*np.pi/8
    return scale*np.float32(q_x)

def phase_extra():
    #加载参数
    # model=onn.Net()
    # model.load_state_dict(torch.load('./saved_model/best.pth'))
    # model.eval()
    # phase1=model.state_dict()['tra.phase']
    # phase=phase1.detach().numpy()
    # scipy.io.savemat("./saved_model/phase.mat",{"phase":phase1})
    phase= scipy.io.loadmat("./saved_model/phase.mat")
    phase=phase["phase"]

    show_output_np(phase)
    

    #模型量化
    q_phase=quantize_tensor(phase)
    q_phase=dequantize_tensor(q_phase)
    show_output_np(q_phase)
    
    return

def evaluate_error():
    args=parameters.my_parameters().get_hyperparameter()

    AL=torch.from_numpy(np.load('./dataset/AL.npy'))
    AR=torch.from_numpy(np.load('./dataset/AR.npy'))
    target_A=torch.stack((AL,AR),0).float()
    delta_phi=torch.from_numpy(np.load('./dataset/delta_phi.npy'))
    amplitude_mask=torch.from_numpy(np.load('./dataset/amplitude_mask.npy'))

    #入射光
    input_images_Total=torch.stack([torch.ones([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_R=torch.stack([torch.zeros([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_Total=input_images_Total/torch.sum(input_images_Total)*1e6

    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/tx_ty.pth'))
    model.eval()
    outputs_L=model(input_images_Total)
    (pre_A,pre_phi)=outputs_L
    
    # show_A=pre_A.detach().numpy()
    # show_phi=pre_phi.detach().numpy()    
    
    #计算整幅图误差
    criterion_sum = torch.nn.MSELoss(reduction='sum')
    criterion_mean = torch.nn.MSELoss(reduction='mean')

    print("*"*20)
    lossA=criterion_mean(pre_A,target_A).float()
    lossphi=criterion_mean(pre_phi,delta_phi).float()
    # print("output1:",torch.sum(pre_A[0,:,:]),torch.max(pre_A[0,:,:]),torch.min(pre_A[0,:,:]))
    # print("output2:",torch.sum(pre_A[1,:,:]),torch.max(pre_A[1,:,:]),torch.min(pre_A[1,:,:]))
    print('整幅图平均损失： A:{:.9f},phi:{:.9f}'.format(lossA,lossphi))

    #计算有掩膜部分误差
    target_A_mask=target_A*amplitude_mask
    delta_phi_mask=delta_phi*amplitude_mask
    pre_A_mask=pre_A*amplitude_mask
    pre_phi_mask=pre_phi*amplitude_mask
    pixel_num=torch.sum(amplitude_mask)
    lossA=criterion_sum(pre_A_mask,target_A_mask).float()/pixel_num
    lossphi=criterion_sum(pre_phi_mask,delta_phi_mask).float()/pixel_num
    print('掩膜覆盖部分平均损失  A:{:.9f},phi:{:.9f}'.format(lossA,lossphi))

    #stocks参量损失
    print("*"*20)
    AL_C=torch.complex(pre_A[1,:,:]*torch.cos(pre_phi).float(),pre_A[1,:,:]*torch.sin(pre_phi).float())
    AR_C=torch.complex(pre_A[0,:,:].float(),torch.zeros_like(pre_A[0,:,:]).float())
    LR=torch.stack((AL_C,AR_C),0)
    stocks_pre=utils.convertLR2stocks(LR)

    AL_C=torch.complex(target_A[1,:,:]*torch.cos(delta_phi).float(),target_A[1,:,:]*torch.sin(delta_phi).float())
    AR_C=torch.complex(target_A[0,:,:].float(),torch.zeros_like(target_A[0,:,:]).float())
    LR=torch.stack((AL_C,AR_C),0)
    stocks_tar =utils.convertLR2stocks(LR)


    loss_s1=criterion_mean(stocks_pre[0,:,:],stocks_tar[0,:,:]).float()
    loss_s2=criterion_mean(stocks_pre[1,:,:],stocks_tar[1,:,:]).float()
    loss_s3=criterion_mean(stocks_pre[2,:,:],stocks_tar[2,:,:]).float()
    print('整图:stocks参量平均损失  s1:{:.9f},s2:{:.9f},s3:{:.9f}'.format(loss_s1,loss_s2,loss_s3))
    stocks_tar=stocks_tar*amplitude_mask
    stocks_pre=stocks_pre*amplitude_mask
    loss_s1=criterion_sum(stocks_pre[0,:,:],stocks_tar[0,:,:]).float()/pixel_num
    loss_s2=criterion_sum(stocks_pre[1,:,:],stocks_tar[1,:,:]).float()/pixel_num
    loss_s3=criterion_sum(stocks_pre[2,:,:],stocks_tar[2,:,:]).float()/pixel_num
    print('掩膜部分:stocks参量平均损失  s1:{:.9f},s2:{:.9f},s3:{:.9f}'.format(loss_s1,loss_s2,loss_s3))


    print("*"*20)
    print("s1 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[0,:,:]),torch.min(stocks_pre[0,:,:]),
                                                torch.max(stocks_tar[0,:,:]),torch.max(stocks_pre[0,:,:])))
    
    print("s2 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[1,:,:]),torch.min(stocks_pre[1,:,:]),
                                                torch.max(stocks_tar[1,:,:]),torch.max(stocks_pre[1,:,:])))
    
    print("s3 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[2,:,:]),torch.min(stocks_pre[2,:,:]),
                                                torch.max(stocks_tar[2,:,:]),torch.max(stocks_pre[2,:,:])))
    
    plt.figure()
    plt.subplot(3,2,1)
    plt.imshow(stocks_tar[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,2)
    plt.imshow(stocks_pre[0,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,3)
    plt.imshow(stocks_tar[1,:,:],cmap='gray',vmin=0)
    plt.subplot(3,2,4)
    plt.imshow(stocks_pre[1,:,:],cmap='gray',vmin=torch.min(stocks_pre[1,:,:]),vmax=torch.max(stocks_pre[1,:,:]))
    plt.subplot(3,2,5)
    plt.imshow(stocks_tar[2,:,:],cmap='gray',vmin=0,vmax=2*torch.pi)
    plt.subplot(3,2,6)
    plt.imshow(stocks_pre[2,:,:],cmap='gray',vmin=0,vmax=2*torch.pi)
    plt.show()


if __name__=='__main__':
    # Sq2=1/np.sqrt(2)
    # E0=torch.complex(torch.tensor([[1,0],[0,1]],dtype=torch.float32),torch.tensor([[0,0],[0,0]],dtype=torch.float32)).reshape([2,1,2])
    
    evaluate_error()
    # show_result()
    # phase_extra()




    
    