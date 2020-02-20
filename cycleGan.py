import networks
import torch
import torch.nn as nn
import Dataloader
import itertools
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class CycleGan(nn.Module):
    def __init__(self,opt):
        super(CycleGan,self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.save_dir = None
        self.loss_names = ['D_A','G_A','cycle_A','D_B','G_B','cycle_B']
        self.model_name = ['G_A','G_B','D_A','D_B']
        self.Train = opt.train  #during train procedure
        if self.Train:
            #generator
            self.G_A = networks.ResNetGenerator(opt.in_fc,opt.out_fc,opt.filter_num)
            self.G_B = networks.ResNetGenerator(opt.in_fc,opt.out_fc,opt.filter_num)
            #discriminator
            self.D_A = networks.Discriminator(opt.in_fc,opt.filter_num)
            self.D_B = networks.Discriminator(opt.in_fc,opt.filter_num)
            #loss
            self.gan_loss = networks.GANLoss(opt.gan_mode).to(self.device)
            self.cycle_loss = nn.L1Loss()
            #optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(),self.G_B.parameters()),lr=opt.lr,betas=(0.5,0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(),self.D_B.parameters()),lr=opt.lr,betas=(0.5,0.999))
            #self.optimizers = [self.optimizer_G,self.optimizer_D]
            #lr
            #self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
    
    def forward(self,real_A,real_B):
        self.real_A = real_A
        self.real_B = real_B
        self.fake_B = self.G_A(real_A).to(self.device)
        self.cycle_A = self.G_B(self.fake_B).to(self.device)
        self.fake_A = self.G_B(real_B).to(self.device)
        self.cycle_B = self.G_A(self.fake_A).to(self.device)
    
    def backward_D_base(self,netD,real,fake):
        pred_real = netD(real)
        loss_real = self.gan_loss(pred_real,True)
        pred_fake = netD(fake)
        loss_fake = self.gan_loss(pred_fake,False)
        loss_D = (loss_real+loss_fake)/2
        loss_D.backward()
        return loss_D

    def backward_D(self):
        '''GAN loss for discriminator A and B'''
        self.loss_D_A = self.backward_D_base(self.D_A,self.real_B,self.fake_B)
        self.loss_D_B = self.backward_D_base(self.D_B,self.real_A,self.fake_A)
        #self.loss_D = (self.loss_D_A+self.loss_D_B)/2

    def backward_G(self):
        '''gan loss and cycle loss for generator A and B'''
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        self.loss_G_A = self.gan_loss(self.D_A(self.fake_B),True)
        self.loss_G_B = self.gan_loss(self.D_B(self.fake_A),True)

        self.cycle_loss_A = self.cycle_loss(self.fake_A,self.real_A)*lambda_A
        self.cycle_loss_B = self.cycle_loss(self.fake_B,self.real_B)*lambda_B

        self.loss_G = (self.loss_G_A+self.loss_G_B+self.cycle_loss_A+self.cycle_loss_B)/4
        self.loss_G.backward(retain_graph=True)
    
    def optimize_parameters(self):
        '''calculate loss and update network weights'''
        # fix discriminator and update generator
        self.set_requires_grad([self.D_A,self.D_B],False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        #fix generator and update discriminator
        self.set_requires_grad([self.D_A,self.D_B],True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_parameters(self):
        if not os.path.isdir(self.opt.save_dir):
            os.mkdir(self.opt.save_dir)
        for name in self.model_name:
            filename = '%s.pth'%name
            save_path = os.path.join(self.opt.save_dir,filename)
            net = getattr(self,name)
            torch.save(net.state_dict(),save_path)
    
    def save_mid_result(self,epoch):
        if not os.path.isdir(self.opt.save_mid_res):
            os.mkdir(self.opt.save_mid_res)
        save_size = len(self.fake_A)
        unloader = transforms.ToPILImage()
        for i in range(save_size):
            filename_A = 'epoch%d_fakeA_%d.png'%(epoch,save_size)
            filename_B = 'epoch%d_fakeB_%d.png'%(epoch,save_size)
            save_pathA = os.path.join(self.opt.save_mid_res,filename_A)
            save_pathB = os.path.join(self.opt.save_mid_res,filename_B)

            imA = self.fake_A.cpu().clone()
            imB = self.fake_B.cpu().clone()
            imA = unloader(imA[0])
            imB = unloader(imB[0])
            imA.save(save_pathA)
            imB.save(save_pathB)

    def get_loss(self):
        return self.loss_D_A,self.loss_D_B,self.loss_G

    
    def load_networks(self):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        d = os.listdir(self.opt.save_dir)
        if len(d) == 0 or '%s.pth'%self.model_name[0] not in d:
            print("There are not any previous parameters being loaded.")
            return
        
        for name in self.model_name:
            load_filename = '%s.pth' % name
            if isinstance(name, str) :
                
                load_path = os.path.join(self.opt.save_dir, load_filename)
                net = getattr(self, name)

                print('loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict)

    # def update_learning_rate(self):
    #     """Update learning rates for all the networks; called at the end of every epoch"""
    #     for scheduler in self.schedulers:
    #         if self.opt.lr_policy == 'plateau':
    #             scheduler.step(self.metric)
    #         else:
    #             scheduler.step()

    #     lr = self.optimizers[0].param_groups[0]['lr']
    #     print('learning rate = %.7f' % lr)