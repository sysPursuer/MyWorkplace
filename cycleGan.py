import networks
import torch
import torch.nn as nn
import Dataloader
import itertools
import os
import numpy as np
from PIL import Image

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
            self.gan_loss = networks.GANLoss(opt.gan_mode)
            self.cycle_loss = nn.L1Loss()
            #optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(),self.G_B.parameters()),lr=opt.lr)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(),self.D_B.parameters()),lr=opt.lr)
        
    
    def forward(self,real_A,real_B):
        self.real_A = real_A
        self.real_B = real_B
        self.fake_B = self.G_A(real_A)
        self.cycle_A = self.G_B(self.fake_B)
        self.fake_A = self.G_B(real_B)
        self.cycle_B = self.G_A(self.fake_A)
    
    def backward_D_base(self,netD,real,fake):
        pred_real = netD(real)
        loss_real = self.gan_loss(pred_real,True)
        pred_fake = netD(fake)
        loss_fake = self.gan_loss(pred_fake,False)
        loss_D = (loss_real+loss_fake)/2
        return loss_D

    def backward_D(self):
        '''GAN loss for discriminator A and B'''
        self.loss_D_A = self.backward_D_base(self.D_A,self.real_B,self.fake_B)
        self.loss_D_B = self.backward_D_base(self.D_B,self.real_A,self.fake_A)
        self.loss_D = (self.loss_D_A+self.D_B)/2
        self.loss_D.backward()

    def backward_G(self):
        '''gan loss and cycle loss for generator A and B'''
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        self.loss_G_A = self.gan_loss(self.D_A(self.fake_B),True)
        self.loss_G_B = self.gan_loss(self.D_B(self.fake_A),True)

        self.cycle_loss_A = self.cycle_loss(self.fake_A,self.real_A)*lambda_A
        self.cycle_loss_B = self.cycle_loss(self.fake_B,self.real_B)*lambda_B

        self.loss_G = (self.loss_G_A+self.loss_G_B+self.cycle_loss_A+self.cycle_loss_B)/4
        self.loss_G.backward()
    
    def optimize_parameters(self):
        '''calculate loss and update network weights'''
        # fix discriminator and update generator
        self.set_requires_grad([self.D_A,self.D_B],False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        #fix generator and update discriminator
        self.set_requires_grad([self.G_A,self.G_A],False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_parameters(self,epoch):
        if not os.path.isdir(self.opt.save_dir):
            os.mkdir(self.opt.save_dir)
        for name in self.model_name:
            filename = '%s_%s.pth'%(epoch,name)
            save_path = os.path.join(self.opt.save_dir,filename)
            net = getattr(self,name)
            torch.save(net.state_dict(),save_path)
    
    def save_mid_result(self):
        if not os.path.isdir(self.opt.save_mid_res):
            os.mkdir(self.opt.save_mid_res)
        save_size = min(5,len(self.fake_A))
        for i in range(save_size):
            filename_A = 'fakeA_%d.png'%save_size
            filename_B = 'fakeB_&d,png'%save_size
            save_pathA = os.path.join(self.opt.save_mid_res,filename_A)
            save_pathB = os.path.join(self.opt.save_mid_res,filename_B)
            imgA = self.tensor2im(self.fake_A[i])
            imgB = self.tensor2im(self.fake_B[i])
            imA = Image.fromarray(imgA)
            imB = Image.fromarray(imgB)
            imA.save(save_pathA)
            imB.save(save_pathB)

    def get_loss(self):
        return self.loss_D,self.loss_G

    def tensor2im(self,input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)