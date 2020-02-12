import argparse
import os
import numpy as np

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNetBlock(nn.Module):
    def __init__(self,in_fc,norm_layer=nn.BatchNorm2d,use_dropout=False):
        super(ResNetBlock,self).__init__()
        block = []
        for i in range(2):
            block += [
                nn.Conv2d(in_fc,in_fc,kernel_size=3,padding=1),
                norm_layer(in_fc),
            ]
            if i==0:
                block+=[nn.LeakyReLU(inplace=True)]
            if use_dropout and i==0:
                block+=[nn.Dropout(0.5)]
        
        self.net = nn.Sequential(*block)
    
    def forward(self,x):
        return self.net(x)+x
        
class ResNetGenerator(nn.Module):
    def __init__(self,in_fc,out_fc,filter_num,block_num=2):
        super(ResNetGenerator, self).__init__()
        model = [
                nn.Conv2d(in_fc,filter_num,kernel_size=5,padding=2),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=True)
        ]
        downsample = 2  #downsample layers
        for i in range(downsample):
            prod = 2**i
            model += [nn.Conv2d(filter_num*prod,filter_num*prod*2,kernel_size=3,stride=2,padding=2),
                    nn.BatchNorm2d(filter_num*prod*2),
                    nn.LeakyReLU(inplace=True)
            ] 
        prod =  2**downsample   #add ResNet blocks
        for i in range(block_num): 
            model += [
                ResNetBlock(filter_num*prod)
            ]
        
        #add upsample layers
        for i in range(downsample):
            prod = 2**(downsample-i)
            model += [nn.ConvTranspose2d(filter_num*prod,int(filter_num*prod/2),
            kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(int(filter_num*prod/2)),
            nn.LeakyReLU(inplace=True)
            ]
        model += [
            nn.Conv2d(filter_num,out_fc,kernel_size=5,padding=1),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*model)

    def forward(self,x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self,in_fc,filter_num,n_layers=2):
        super(Discriminator,self).__init__()
        model = []
        model += [
            nn.Conv2d(in_fc,filter_num,kernel_size=3,padding=1,stride=2),
            nn.LeakyReLU(inplace=True)
        ]
        prod=1
        prod_prev=1
        for i in range(1,n_layers+1):
            prod_prev = prod
            prod = min(2**i,8)
            model+=[
                nn.Conv2d(filter_num*prod_prev,filter_num*prod,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(filter_num*prod),
                nn.LeakyReLU(inplace=True)
            ]
        model +=[
            nn.Conv2d(filter_num*prod,1,kernel_size=3,padding=1)
        ]
        self.net = nn.Sequential(*model)

    def forward(self,x):
        return self.net(x)

class GANLoss(nn.Module):
    def __init__(self,gan_mode,target_is_real=1.0,target_is_fake=0.0):
        super(GANLoss,self).__init__()
        self.register_buffer('real_label',torch.tensor(target_is_real))
        self.register_buffer('fake_label',torch.tensor(target_is_fake))

        self.gan_mode = gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode== 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self,prediction,target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self,prediction,target_is_real):
        if self.gan_mode in ['lsgan','vanilla']:
            target_tensor = self.get_target_tensor(prediction,target_is_real)
            loss = self.loss(prediction,target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss