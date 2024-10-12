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
#显示结果
def show_AF():
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

def error_showStocks():
    args=parameters.my_parameters().get_hyperparameter()

    AL=torch.from_numpy(np.load('./dataset/AL.npy'))
    AR=torch.from_numpy(np.load('./dataset/AR.npy'))
    target_A=torch.stack((AL,AR),0).float()
    delta_phi=torch.from_numpy(np.load('./dataset/delta_phi.npy'))
    amplitude_mask=torch.from_numpy(np.load('./dataset/amplitude_mask.npy'))

    AL_C=torch.complex(target_A[0,:,:].float(),torch.zeros_like(target_A[0,:,:]).float())
    AR_C=torch.complex(target_A[1,:,:]*torch.cos(delta_phi).float(),target_A[1,:,:]*torch.sin(delta_phi).float())
    LR=torch.stack((AL_C,AR_C),0)
    I_tar,stocks_tar =utils.convertLR2stocks(LR)
    total_energy=torch.sum(I_tar)
    
    #入射光，入射光的能量应该设置为多少？目标图像能量是固定的，而输出图像能量是正比于入射光能量的。
    input_images_Total=torch.stack([torch.ones([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()

    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.eval()
    outputs_L=model(input_images_Total)
    (pre_A,pre_phi)=outputs_L
    
    
    #计算整幅图误差
    criterion_sum = torch.nn.MSELoss(reduction='sum')
    criterion_mean = torch.nn.MSELoss(reduction='mean')

    #stocks参量损失
    print("*"*20)
    AL_C=torch.complex(pre_A[0,:,:].float(),torch.zeros_like(pre_A[0,:,:]).float())
    AR_C=torch.complex(pre_A[1,:,:]*torch.cos(pre_phi).float(),pre_A[1,:,:]*torch.sin(pre_phi).float())
    
    LR=torch.stack((AL_C,AR_C),0)
    I_pre,stocks_pre=utils.convertLR2stocks(LR)
    I_pre=I_pre /torch.sum(I_pre) *total_energy

    stocks_tar=stocks_tar*amplitude_mask
    stocks_pre=stocks_pre*amplitude_mask
    pixel_num=torch.sum(amplitude_mask)
    loss_I =criterion_sum(I_pre,I_tar).float()/1e6
    loss_s1=criterion_sum(stocks_pre[0,:,:],stocks_tar[0,:,:]).float()/pixel_num
    loss_s2=criterion_sum(stocks_pre[1,:,:],stocks_tar[1,:,:]).float()/pixel_num
    loss_s3=criterion_sum(stocks_pre[2,:,:],stocks_tar[2,:,:]).float()/pixel_num
    print('掩膜部分:stocks参量平均损失  s1:{:.9f},s2:{:.9f},s3:{:.9f},I:{:.9f}'.format(loss_s1,loss_s2,loss_s3,loss_I))

    print("*"*20)
    print("s1 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[0,:,:]),torch.min(stocks_pre[0,:,:]),
                                                torch.max(stocks_tar[0,:,:]),torch.max(stocks_pre[0,:,:])))
    
    print("s2 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[1,:,:]),torch.min(stocks_pre[1,:,:]),
                                                torch.max(stocks_tar[1,:,:]),torch.max(stocks_pre[1,:,:])))
    
    print("s3 min:{}, {}   ; max:{}, {}".format(torch.min(stocks_tar[2,:,:]),torch.min(stocks_pre[2,:,:]),
                                                torch.max(stocks_tar[2,:,:]),torch.max(stocks_pre[2,:,:])))
    print( "I min:{}, {}   ; max:{}, {}".format(torch.min(I_tar),torch.min(I_pre),
                                                torch.max(I_tar),torch.max(I_pre)))
    
    stocks_pre=stocks_pre.detach().numpy()
    I_pre=I_pre.detach().numpy()
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(stocks_tar[0,:,:],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(stocks_pre[0,:,:],cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(stocks_tar[1,:,:],cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(stocks_pre[1,:,:],cmap='gray')

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(stocks_tar[2,:,:],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(stocks_pre[2,:,:],cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(I_tar,cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(I_pre,cmap='gray',vmin=0,vmax=1)
    plt.show()


#显示纯几何相位全息图，输入：np格式旋转角度，显示菲涅尔衍射结果图片
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
    Al = np.fft.ifftshift(np.fft.ifft2(k_space_L))
    Ar = np.fft.ifftshift(np.fft.ifft2(k_space_R))

    res_angle=np.angle(Ar)-np.angle(Al)
    res_angle=res_angle%(2*torch.pi)

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Al')
    plt.imshow(abs(Al),cmap='gray',vmin=0)
    plt.subplot(2,2,2)
    plt.title('Ar')
    plt.imshow(abs(Ar),cmap='gray',vmin=0)
    plt.subplot(2,2,3)
    plt.title('delta_angle')
    plt.imshow(res_angle,cmap='gray',vmin=0,vmax=np.pi*2)
    plt.subplot(2,2,4)
    plt.title('phase')
    plt.imshow(phase,cmap='gray',vmin=0,vmax=np.pi*2)
    plt.show()
    return


#将模型参数保存为mat格式
#phase：旋转角R【0，2*pi】，量化为8阶，为0-7
#
def phase_extra():
    #加载参数
    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.eval()
    SD1=model.state_dict()
    phase1=SD1['tra.phase']
    phase=phase1.detach().numpy()
    delta=SD1['tra.delta']
    delta.data.clamp_(0,np.pi)
    delta=delta.detach().numpy()
    
    #模型量化
    q_delta=utils.quantize_tensor(delta)
    scale_phase=2*np.pi/65536
    q_phase=(phase/scale_phase).round()%65536
    
    # print("q_phase,min:{},max:{}".format(np.min(q_phase),np.max(q_phase)))
    print("q_tx,min:{},max:{}".format(np.min(q_delta),np.max(q_delta)))
    #范围【0，2*pi】
    scipy.io.savemat("./saved_model/PD.mat",{"phase":q_phase,"delta":q_delta})
    
    #测试量化效果
    args=parameters.my_parameters().get_hyperparameter()
    input_images_Total=torch.stack([torch.ones([args.img_size,args.img_size]),torch.ones([args.img_size,args.img_size])],0).float()
    input_images_Total=input_images_Total/torch.sum(input_images_Total)*1e6

    SD1=model.state_dict()
    new_delta=utils.dequantize_tensor(q_delta)
    SD1['tra.delta']=torch.from_numpy(new_delta)

    new_phase=q_phase*scale_phase
    SD1['tra.phase']=torch.from_numpy(new_phase)
    model.load_state_dict(SD1)
    torch.save(model.state_dict(),'./saved_model/test.pth')
    return

def phase_extra_onlyR():
    #加载参数
    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.eval()
    SD1=model.state_dict()
    phase1=SD1['tra.phase']
    phase=phase1.detach().numpy()
    # phase= scipy.io.loadmat("./saved_model/phase.mat")
    # phase=phase["phase"]
    
    #模型量化
    scale_phase=2*np.pi/65536
    q_phase=(phase/scale_phase).round()%65536
    print("q_phase,min:{},max:{}".format(np.min(q_phase),np.max(q_phase)))
    
    #范围【0，2*pi】
    scipy.io.savemat("./saved_model/only_R.mat",{"phase":q_phase})
    new_phase=q_phase*scale_phase
    show_output_np(new_phase)
    return
    

if __name__=='__main__':
    # Sq2=1/np.sqrt(2)
    # E0=torch.complex(torch.tensor([[1,0],[1,0]],dtype=torch.float32),torch.tensor([[0,-1],[0,1]],dtype=torch.float32)).reshape([2,1,2])*Sq2
    # print(utils.convertLR2stocks(E0))
    
    # show_AF()
    # phase_extra()
    error_showStocks()
    # phase_extra_onlyR()

