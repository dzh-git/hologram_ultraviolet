import torch
import numpy as np
import parameters
import random


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

class TransmissionLayer(torch.nn.Module):
    def __init__(self):
        super(TransmissionLayer, self).__init__()
        self.args=parameters.my_parameters().get_hyperparameter()
        self.actual_situation = parameters.my_parameters().get_actualparameter()
        self.phase = torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        #空间复用
        # mask_L2R=torch.from_numpy(np.fromfunction(lambda i, j: (i+j)%2,shape=(self.args.img_size, self.args.img_size), dtype=np.int16))
        # mask_R2L=1-mask_L2R
        # self.grometry_mask=torch.nn.Parameter(torch.stack((mask_L2R,mask_R2L),0).float(),requires_grad=False)
        # self.cross_talk=torch.nn.Parameter(torch.stack((-mask_R2L,-mask_L2R),0).float(),requires_grad=False)

        self.grometry_mask=torch.nn.Parameter(
            torch.stack([torch.ones([self.args.img_size,self.args.img_size]),-torch.ones([self.args.img_size,self.args.img_size])],0)
            .float(),requires_grad=False)

    def forward(self, x):
        if self.actual_situation.manufacturing_error:
            mask =self.phase + torch.from_numpy(np.random.random(size=[self.args.img_size
                    ,self.args.img_size]).astype('float32')).cuda()*random.choice([1,-1])*2
            mask=torch.complex(torch.cos(mask), torch.sin(mask))
        else:
            new_phase=torch.mul(self.grometry_mask,self.phase)
            mask=torch.complex(torch.cos(new_phase), torch.sin(new_phase))
        x=torch.mul(x,mask)
        # x=torch.mul(torch.mul(x,self.grometry_mask),mask)
        # x_cross_talk=torch.mul(torch.mul(x,self.cross_talk),mask)
        # x=x+x_cross_talk
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
        res_angle=torch.angle(x[1,:,:]-x[0,:,:])
        res_angle=(res_angle+2*torch.pi)%(2*torch.pi)

        x_abs=abs(x) ; x_energy=x_abs*x_abs
        ss=torch.sum(x_energy,dim=(0,1,2)).unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2) #归一化
        x_energy=x_energy.div(ss)*2
        return (x_energy,res_angle)

if __name__=="__main__":
    # x=torch.randn((2,1,51,51))
    # detect_region(x)
    pp=parameters.my_parameters().get_hyperparameter()
    print(pp.img_size)

