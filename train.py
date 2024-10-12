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


def load_img(args,path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img0=cv2.resize(img0,dsize=(args.img_size,args.img_size))
    #归一化，总能量为1
    img0=img0/np.sum(img0)
    return img0
'''
输入：tensor[2,W,H]，第一个通道和第二个通道代表右旋和左旋。
输出：tensor[2,W,H]，由于几何相位，输出的第0个通道代表左旋，第1个通道代表右旋
相位差delta_phi表示 右旋-左旋
'''
def main(args):
    #模型保存路径
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    AL=torch.from_numpy(np.load('./dataset/AL.npy'))
    AR=torch.from_numpy(np.load('./dataset/AR.npy'))
    target_A=torch.stack((AL,AR),0).float()
    delta_phi=torch.from_numpy(np.load('./dataset/delta_phi.npy')).float()
    amplitude_mask=np.load('./dataset/amplitude_mask.npy')
    background_mask=torch.from_numpy(1-amplitude_mask).float()
    amplitude_mask=torch.from_numpy(amplitude_mask).float()


    if device !='cpu':
        target_A=target_A.cuda()
        delta_phi=delta_phi.cuda()
        amplitude_mask=amplitude_mask.cuda()
        background_mask=background_mask.cuda()
    delta_phi=delta_phi *amplitude_mask
    target_A=target_A   *amplitude_mask
    pixel_num=torch.sum(amplitude_mask)
    target_A=target_A   #/torch.sum(amplitude_mask)*1e6
    #目标的stocks参量
    AL_C=torch.complex(target_A[0,:,:].float(),torch.zeros_like(target_A[0,:,:]).float())
    AR_C=torch.complex(target_A[1,:,:]*torch.cos(delta_phi).float(),target_A[1,:,:]*torch.sin(delta_phi).float())
    
    LR=torch.stack((AL_C,AR_C),0)
    I_tar,stocks_tar =utils.convertLR2stocks(LR)
    I_tar=I_tar*amplitude_mask  
    total_energy=torch.sum(I_tar)
    stocks_tar=stocks_tar*amplitude_mask
    

    #线偏光入射
    train_images=torch.ones(size=[2,args.img_size,args.img_size]) if device=='cpu'  else torch.ones(size=[2,args.img_size,args.img_size]).cuda()
    train_images=train_images
    

    model=onn.Net()
    model.load_state_dict(torch.load(r'./saved_model/best.pth'))
    model.to(device)
    
    criterion = torch.nn.MSELoss(reduction='sum') if device == "cpu" else torch.nn.MSELoss(reduction='sum').cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    early_stopping=EarlyStop()
    for epoch in range(args.num_epochs):
        model.train()
        outputs=model(train_images)
        (pre_A,pre_phi)=outputs

        #振幅相位损失
        # lossA=criterion(pre_A,target_A).float()
        # lossphi=criterion(pre_phi,delta_phi).float()
        # loss1=lossA+lossphi
        
        #stocks参量损失
        AL_C=torch.complex(pre_A[0,:,:].float(),torch.zeros_like(pre_A[0,:,:]).float())
        AR_C=torch.complex(pre_A[1,:,:]*torch.cos(pre_phi).float(),pre_A[1,:,:]*torch.sin(pre_phi).float())
        LR=torch.stack((AL_C,AR_C),0)
        I_pre,stocks_pre=utils.convertLR2stocks(LR)
        I_pre=I_pre /torch.sum(I_pre) *total_energy #能量归一化
        
        
        
        stocks_pre=stocks_pre*amplitude_mask

        # loss_stocks=criterion(stocks_pre,stocks_tar).float()/pixel_num
        loss_I1=criterion(I_pre,I_tar).float()/1e6  #损失：归一化后，整图光强分布误差
        loss_s1=criterion(stocks_pre[0,:,:],stocks_tar[0,:,:]).float()/pixel_num
        loss_s2=criterion(stocks_pre[1,:,:],stocks_tar[1,:,:]).float()/pixel_num
        loss_s3=criterion(stocks_pre[2,:,:],stocks_tar[2,:,:]).float()/pixel_num
        total_loss=loss_s1*1.2 + loss_s2*1.2 + loss_s3*1.2 +loss_I1

        if torch.isnan(total_loss):
            print('loss is nan , break')
            break
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch%50==0:
            print('loss_I1:{:.9f},loss_s1:{:.9f},loss_s2:{:9f},loss_s3:{:9f}'.format(loss_I1,loss_s1,loss_s2,loss_s3))

        early_stopping(-total_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(model.state_dict(),'./saved_model/last.pth')


if __name__=='__main__':
    args=parameters.my_parameters().get_hyperparameter()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    main(args)
    
