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
    
    AL=load_img(args,'./dataset/left.png')
    AL=torch.from_numpy(AL).float() if device=='cpu' else torch.from_numpy(AL).float().cuda()
    AR=load_img(args,'./dataset/right.png')
    AR=torch.from_numpy(AR).float() if device=='cpu' else torch.from_numpy(AR).float().cuda()
    target_A=torch.stack((AL,AR),0)
    delta_phi=torch.zeros((args.img_size,args.img_size))
    delta_phi[args.img_size//2:,:args.img_size//2]=torch.pi/2 ; delta_phi[args.img_size//2:,args.img_size//2:]=torch.pi
    delta_phi[:args.img_size//2,args.img_size//2:]=torch.pi*3/2
    if device !='cpu':
        delta_phi=delta_phi.cuda()
    #线偏光入射
    pixel_num=args.img_size*args.img_size
    train_images=torch.ones(size=[2,args.img_size,args.img_size])/pixel_num if device=='cpu'  else (torch.ones(size=[2,args.img_size,args.img_size])/pixel_num).cuda()

    model=onn.Net()
    model.load_state_dict(torch.load('./saved_model/best.pth'))
    model.to(device)
    
    criterion = torch.nn.MSELoss(reduction='sum') if device == "cpu" else torch.nn.MSELoss(reduction='sum').cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    early_stopping=EarlyStop()
    for epoch in range(args.num_epochs):
        model.train()
        outputs=model(train_images)
        (pre_A,pre_phi)=outputs

        phi_norm=1/(2*torch.pi*args.img_size**2)
        lossA=criterion(pre_A,target_A).float()
        lossphi=criterion(pre_phi,delta_phi).float()*phi_norm
        total_loss=lossA+lossphi*1e-3
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch%50==0:
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
    
