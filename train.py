import torch
import cycleGan
import Dataloader
from argparse import ArgumentParser
import torch.utils.data as Data

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_options():
    parser = ArgumentParser()
    #network parameters
    parser.add_argument('--in_fc',type=int,default=3)
    parser.add_argument('--out_fc',type=int,default=3)
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument('--filter_num',type=int,default=16)
    parser.add_argument('--trainA_path',type=str,default='./horse2zebra/trainA')
    parser.add_argument('--trainB_path',type=str,default='./horse2zebra/trainB')
    parser.add_argument('--save_dir',type=str,default='save_parameter')
    parser.add_argument('--save_mid_res',type=str,default='mid_result')
    parser.add_argument('--max_size',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=1)

    parser.add_argument('--save_iter_freq',type=int,default=80)
    parser.add_argument('--save_mid_epoch',type=int,default=3)

    parser.add_argument('--lr',type=float,default=0.0002)
    parser.add_argument('--gan_mode',type=str,default='lsgan')
    parser.add_argument('--epoch_num',type=int,default=5000)
    parser.add_argument('--lambda_A',type=float,default=0.3)
    parser.add_argument('--lambda_B',type=float,default=0.3) 
    parser.add_argument('--gpu_ids',type=str,default='0')
    parser.add_argument('--lr_policy',type=str,default='linear')
    parser.add_argument('--niter_decay',type=int,default=100,help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_decay_iters',type=int,default=50,help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    return parser.parse_args()

def train():
    options = get_options()
    dataloader = Dataloader.get_dataloader(options.trainA_path,options.trainB_path,options.max_size,options.batch_size)
    cyclegan_model = cycleGan.CycleGan(options)
    cyclegan_model.to(DEVICE)
    cyclegan_model.load_networks()
    total_iters = 0
    try:
        for epoch in range(options.epoch_num):
            for step,(real_A,real_B) in enumerate(dataloader):
                total_iters += options.batch_size
                cyclegan_model.forward(real_A,real_B)
                cyclegan_model.optimize_parameters()
                # if total_iters % options.save_iter_freq==0:
                #     cyclegan_model.save_parameters()
                lossD_A,lossD_B,lossG = cyclegan_model.get_loss()
                if step%10==0:
                    print('Epoch:',epoch,' | step:',step,' | lossD_A:',lossD_A.item(),' | lossD_B:',lossD_B.item(),' | lossG:',lossG.item())
            cyclegan_model.save_parameters()
            if epoch%options.save_mid_epoch==0:
                cyclegan_model.save_mid_result(epoch)
            #update lr after every epoch
            #cyclegan_model.update_learning_rate()
    finally:
        cyclegan_model.save_parameters()
if __name__ == '__main__':
    train()