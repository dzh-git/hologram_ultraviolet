import torch
import numpy as np
import parameters
import random
import torch.nn as nn


class DiffractiveLayer(torch.nn.Module):
    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        args=parameters.my_parameters().get_hyperparameter()
        actual_situation = parameters.my_parameters().get_actualparameter()
        distance=actual_situation.distance
        wave_length=actual_situation.wave_length
        screen_length=actual_situation.screen_length
        wave_num=2*3.14159/wave_length

        #dx表示衍射层像素大小，1/2dx为最大空间采样频率，不改这个
        point_num=args.img_size; dx=screen_length/point_num

        fx_list=np.arange(-1/(2*dx),1/(2*dx),1/screen_length)
        phi = np.fromfunction(
            lambda i, j: 1-(np.square(wave_length*fx_list[i])+np.square(wave_length*fx_list[j])),
            shape=(point_num, point_num), dtype=np.int16).astype(np.complex64)
        H = np.exp(1.0j * wave_num * distance*np.sqrt(phi))
        self.H=torch.fft.fftshift(torch.complex(torch.from_numpy(H.real),torch.from_numpy(H.imag)), dim=(0,1))
        self.H = torch.nn.Parameter( self.H, requires_grad=False)
        
    #在频域进行计算，看信息光学
    def forward(self, waves):
        temp = torch.fft.fft2(torch.fft.fftshift(waves, dim=(1,2)), dim=(1,2))
        k_space=temp.mul(self.H)
        x_space = torch.fft.ifftshift(torch.fft.ifft2(k_space, dim=(1,2)), dim=(1,2))
        return x_space

class DiffraFouriourLayer(torch.nn.Module):
    def __init__(self):
        super(DiffraFouriourLayer, self).__init__()
        
    #在频域进行计算，看信息光学
    def forward(self, waves):
        k_space=torch.fft.fft2(torch.fft.fftshift(waves, dim=(1,2)), dim=(1,2))
        x_space = torch.fft.ifftshift(k_space, dim=(1,2))
        return x_space
 

class TransmissionLayer(torch.nn.Module):
    def __init__(self):
        super(TransmissionLayer, self).__init__()
        self.args=parameters.my_parameters().get_hyperparameter()
        self.actual_situation = parameters.my_parameters().get_actualparameter()
        self.phase = torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        #假如相位差不为pi,且tx，ty的膜为1,只引入相位差
        self.txp=torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        self.typ=torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        #左右旋
        self.grometry_mask=torch.nn.Parameter(
            torch.stack([torch.ones([self.args.img_size,self.args.img_size]),-torch.ones([self.args.img_size,self.args.img_size])],0)
            .float(),requires_grad=False)
        #
        #空间复用
        # mask_L2R=torch.from_numpy(np.fromfunction(lambda i, j: (i+j)%2,shape=(self.args.img_size, self.args.img_size), dtype=np.int16))
        # mask_R2L=1-mask_L2R
        # self.grometry_mask=torch.nn.Parameter(torch.stack((mask_L2R,mask_R2L),0).float(),requires_grad=False)
        # self.cross_talk=torch.nn.Parameter(torch.stack((mask_R2L,mask_L2R),0).float(),requires_grad=False)

    def forward(self, x):
        if self.actual_situation.manufacturing_error:
            mask =self.phase + torch.from_numpy(np.random.random(size=[self.args.img_size
                    ,self.args.img_size]).astype('float32')).cuda()*random.choice([1,-1])*2
        new_phase=torch.mul(self.grometry_mask,self.phase)
        mask=torch.complex(torch.cos(new_phase), torch.sin(new_phase))
        #假如相位差不为pi,且tx，ty的膜为1,只引入相位差
        co=(torch.complex(torch.cos(self.txp), torch.sin(self.txp))+torch.complex(torch.cos(self.typ), torch.sin(self.typ)))/2
        cross=(torch.complex(torch.cos(self.txp), torch.sin(self.txp))-torch.complex(torch.cos(self.typ), torch.sin(self.typ)))/2
        temp=torch.mul(x,mask)  
        #temp的0通道是右旋圆偏光，输入的x的0通道是左旋圆偏光，输出的x的0通道是右旋圆偏光
        output=torch.zeros_like(x)
        output[0,:,:]=co*x[1,:,:]+cross*temp[0,:,:]
        output[1,:,:]=co*x[0,:,:]+cross*temp[1,:,:]
        return output

        #空间复用，考虑或不考虑串扰
        # mask=torch.complex(torch.cos(self.phase), torch.sin(self.phase))
        # out1=torch.mul(torch.mul(x,self.grometry_mask),mask)
        # cross_x=torch.mul(torch.mul(x,self.cross_talk),-mask)
        # x=out1+cross_x
        return x

class DTLayer(torch.nn.Module):
    def __init__(self):
        super(DTLayer,self).__init__()
        self.dif=DiffractiveLayer()
        self.tra=TransmissionLayer()

    def forward(self,x):
        x=self.dif(x)
        x=self.tra(x)
        return x

class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, num_layers=1):
        super(Net, self).__init__()
        self.tra=TransmissionLayer()
        self.dif=DiffractiveLayer()
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.actual_situation = parameters.my_parameters().get_actualparameter()
        self.args=parameters.my_parameters().get_hyperparameter()
        
    def forward(self, x):
        # x (200, 200)  torch.complex64
        #表示斜入射
        if self.actual_situation.oblique_incidence:
            #光源随机角度+-4°
            random_thetax=np.random.random()*random.choice([1,-1])*2
            random_thetay=np.random.random()*random.choice([1,-1])*2
            wave_length=self.actual_situation.wave_length
            screen_length=self.actual_situation.screen_length
            wave_num=2*3.14159/wave_length
            dx=screen_length/self.args.img_size
            x_list=np.arange(-0.5*screen_length,0.5*screen_length,dx)

            tilt_phase = torch.from_numpy(np.fromfunction(
            lambda i, j: wave_num*(x_list[i]*np.sin(random_thetax)+x_list[j]*np.sin(random_thetay)),
            shape=(self.args.img_size, self.args.img_size), dtype=np.int16).astype(np.float32))
            mask=torch.complex(torch.cos(tilt_phase), torch.sin(tilt_phase)).cuda()
            x=torch.mul(x,mask)
        x=self.tra(x)
        x=self.dif(x)
        # return x
    
        res_angle=torch.angle(x[1,:,:])-torch.angle(x[0,:,:])
        res_angle=res_angle%(2*torch.pi)

        x_abs=abs(x) ; #x_energy=x_abs*x_abs
        # x_energy=x_energy/torch.sum(x_energy)
        return (x_abs,res_angle)

if __name__=="__main__":
    # x=torch.randn((2,1,51,51))
    # detect_region(x)
    pp=parameters.my_parameters().get_hyperparameter()
    print(pp.img_size)

