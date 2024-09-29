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
输入：tensor[2,W,H]，第一个通道和第二个通道代表左旋和右旋。
输出：tensor[2,W,H]，由于几何相位，输出的第一个通道代表右旋，第二个通道代表左旋
'''
def main(args):
    #模型保存路径
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    path1='./dataset/newleft.png';path2='./dataset/newright.png'
    AL=load_img(args,path1)
    AL=torch.from_numpy(AL).float() if device=='cpu' else torch.from_numpy(AL).float().cuda()
    AR=load_img(args,path2)
    AR=torch.from_numpy(AR).float() if device=='cpu' else torch.from_numpy(AR).float().cuda()
    phase_mask=utils.calc_phaseMask(args,path1,path2)
    target_A=torch.stack((AL,AR),0)

    delta_phi=torch.zeros((args.img_size,args.img_size))
    delta_phi[:args.img_size//2,:args.img_size//2]=0 ; delta_phi[:args.img_size//2,args.img_size//2:]=torch.pi
    delta_phi[args.img_size//2:,:args.img_size//2]=torch.pi*3/2;delta_phi[args.img_size//2:,args.img_size//2:]=torch.pi/2
    if device !='cpu':
        delta_phi=delta_phi.cuda()
        phase_mask=phase_mask.cuda()
    delta_phi=delta_phi*phase_mask
    target_A=target_A*phase_mask
    target_A=target_A/2*1e6

    # target_Eab=torch.load('./dataset/Eab.pt')
    # target_Exy=torch.load('./dataset/Exy.pt')
    # target_Elr=torch.load('./dataset/Elr.pt')
    # gd_mask=torch.zeros([1000,1000])
    # gd_mask[350:650,350:650]=torch.ones([300,300])

    # if device !='cpu':
    #     target_Eab=target_Eab.cuda()
    #     target_Exy=target_Exy.cuda()
    #     target_Elr=target_Elr.cuda()
    #     gd_mask=gd_mask.cuda()

    #线偏光入射
    pixel_num=args.img_size*args.img_size
    train_images=torch.ones(size=[2,args.img_size,args.img_size])/pixel_num if device=='cpu'  else (torch.ones(size=[2,args.img_size,args.img_size])/pixel_num).cuda()
    total_energy=torch.sum(train_images)
    train_images=train_images/total_energy*1e6

    model=onn.Net()
    # model.load_state_dict(torch.load(r'./saved_model/best.pth'))
    model.to(device)
    
    criterion = torch.nn.MSELoss(reduction='sum') if device == "cpu" else torch.nn.MSELoss(reduction='sum').cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    early_stopping=EarlyStop()
    for epoch in range(args.num_epochs):
        model.train()
        outputs=model(train_images)
        (pre_A,pre_phi)=outputs

        pre_A=pre_A*phase_mask
        pre_phi=pre_phi*phase_mask
        lossA=criterion(pre_A,target_A).float()
        lossphi=criterion(pre_phi,delta_phi).float()
        total_loss=lossA+lossphi*0.25
        if torch.isnan(total_loss):
            print('loss is nan , break')
            break
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch%50==0:
            print(torch.sum(pre_A))
            print('A:{:.9f},phi:{:.9f},total:{:.9f}'.format(lossA,lossphi,total_loss))
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
    
