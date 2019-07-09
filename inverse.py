# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR
import pdb
import copy
from resnet import resnet18

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset',  help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    # parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=5)
    parser.add_argument('--bs', type=int, default=64, help='input batch size')
    # parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
    parser.add_argument('--gamma', type=float, default=0.4, help='gamma for learning rate, default=0.4')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--debug', action='store_true', help='enable debug')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--cp', default='', help="path to checkpoint (to continue training or evaluation)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--eval', help='Only do evaluation', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    args = parser.parse_args()
    return args

# if dim=None, can be used on a pytorch tensor as well as a numpy array
def calc_tensor_stats(arr, do_print=False, arr_name=None, dim=None):
    if dim is None:
        arr_min = arr.min()
        arr_max = arr.max()
        arr_mean = arr.mean()
        arr_std  = arr.std()
        stats = arr_min.item(), arr_max.item(), arr_mean.item(), arr_std.item()
    else:
        arr_min = arr.min(dim=dim)[0]
        arr_max = arr.max(dim=dim)[0]
        arr_mean = arr.mean(dim=dim)
        arr_std  = arr.std(dim=dim)
        stats = arr_min, arr_max, arr_mean, arr_std
    
    if do_print:
        print("%s: min %.3f max %.3f mean %.3f std %.3f" %((arr_name,) + stats) )
    return stats

sim_train_folder = './sim-train2f'
sim_val_folder = './sim-val2f'
model_folder = './model2f'
args = parse_args()
print(args)

for path in (sim_train_folder, sim_val_folder, model_folder):
    try:
        os.makedirs(path)
    except OSError:
        pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)

class NDTDataset(Dataset):
    def __init__(self, resp_folders, index_range,
                 truth_folder, pix_offset, transforms, perturb_std=0, perturb_sometimes=0.5):
        self.transforms = transforms
        all_resps = [ [] for i in resp_folders ]
        truths = []
        index_lb, index_ub = index_range
        
        resp_folder0 = resp_folders[0]
        for filename in os.listdir(resp_folder0):
            file_index, _ = filename.split(".")
            file_index = int(file_index)
            if file_index < index_lb or file_index >= index_ub:
                continue
                
            truths.append(  cv2.imread( os.path.join(truth_folder, filename), cv2.IMREAD_GRAYSCALE))
            for i, resp_folder in enumerate(resp_folders):
                all_resps[i].append( cv2.imread( os.path.join(resp_folder, filename), cv2.IMREAD_GRAYSCALE))

        truths = np.expand_dims( np.array(truths), axis=3 )
        print("truths shape: %s" %(str(truths.shape)))

        for i, resps in enumerate(all_resps):
            resps = all_resps[i]
            all_resps[i] = np.expand_dims( np.array(resps), axis=3 )
            print("response-%d shape: %s" %(i, str(all_resps[i].shape)))
            
        # calc_tensor_stats(truths, do_print=True, arr_name='truths')

        self.all_resps = all_resps
        
        self.truths = truths
        self.pix_offset = pix_offset
        self.perturb_std = perturb_std
        self.perturb_sometimes = perturb_sometimes
        
    def __len__(self):
        return len(self.truths)

    def __getitem__(self, idx):
        truth = self.truths[idx]
        all_resp_tensors = []
        for resps in self.all_resps:
            resp = resps[idx]
            resp_tensor = self.transforms(resp) + self.pix_offset
        
            if self.perturb_std > 0 and np.random.binomial(1, self.perturb_sometimes) == 1:
                resp_tensor += torch.randn(resp_tensor.shape) * self.perturb_std
        
            all_resp_tensors.append(resp_tensor)
        
        all_resp_tensors = torch.cat(all_resp_tensors, 0)
        truth_tensor = self.transforms(truth) + self.pix_offset
        return all_resp_tensors, truth_tensor

pix_offset = -0.5
resp_folders =  [
                    "/home/shaohua/GAN/conv_dataset/real_groundtruth1", 
                    "/home/shaohua/GAN/conv_dataset/real_groundtruth3", 
                    "/home/shaohua/GAN/conv_dataset/real_groundtruth6", 
                    "/home/shaohua/GAN/conv_dataset/imag_groundtruth1",
                    "/home/shaohua/GAN/conv_dataset/imag_groundtruth3",
                    "/home/shaohua/GAN/conv_dataset/imag_groundtruth6"
                 ]
                      
ndt_trainset = NDTDataset(resp_folders, (0, 16000),
                         "/home/shaohua/GAN/conv_dataset/data12f2",
                          pix_offset, transforms.ToTensor(), 0.0)

ndt_validset = NDTDataset(resp_folders, (16000, 20000),
                          "/home/shaohua/GAN/conv_dataset/data12f2",
                          pix_offset, transforms.ToTensor(), 0)
                            
# all_indices = np.arange(len(ndt_trainset))
# train_indices, val_indices = train_test_split(all_indices, test_size=0.3, random_state=33)
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler   = SubsetRandomSampler(val_indices)

train_loader = DataLoader(ndt_trainset, batch_size=args.bs, shuffle=True, num_workers=int(args.workers))
val_loader   = DataLoader(ndt_validset, batch_size=args.bs, num_workers=int(args.workers))
num_img_channels = 1

print("%d training data" %(len(train_loader) * args.bs))
print("%d validation data" %(len(val_loader) * args.bs))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
nc = len(resp_folders)                                       

class Flatten(nn.Module):
    def forward(self, input):
        tens = input.view(input.size(0), -1,1,1)
        tens = (tens.reshape(tens.size(0),1,40,12))
        return tens

class InvGenerator(nn.Module):
    def __init__(self, ngpu):
        super(InvGenerator, self).__init__()
        self.ngpu = ngpu
        C1 = C2 = C3 = C4 = 5
        self.encoder = nn.Sequential(
            # input is (nc=1) x 40 x 40
            nn.Conv2d(nc, ndf * C1, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * C1),
            # nn.ReLU(True),
            nn.LeakyReLU(0.3, inplace=True),
            # state size. (ndf * 2) x 35 x 35
            nn.Conv2d(ndf * C1, ndf * C2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * C2),
            # nn.ReLU(True), 
            nn.LeakyReLU(0.3, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * C2, ndf * C3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * C3),
            # nn.ReLU(True),
            nn.LeakyReLU(0.3, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * C3, ndf * C4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * C4),
            # nn.ReLU(True),
            nn.LeakyReLU(0.3, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * C4, nz, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.ReLU(True),
            # state size. (ndf*8) x 3 x 3
            # nn.Conv2d(ndf * 4, nz, 3, 1, 0, bias=False),
            # 480 * 1 * 1
        )
        # self.encoder = resnet18(no_fc=True)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 3, 1, 0, bias=False),          # 13
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # image size: (ngf*8) x 3 x 3
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, (2,1), 1, bias=False),  # 16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # image size: (ngf*4) x 5 x 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),      # 19
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # image size: (ngf*2) x 10 x 6
            nn.ConvTranspose2d(ngf * 2, ngf, (4,3), (2,1), 1, bias=False),  # 22
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # image size: (ngf*2) x 20 x 5
            nn.ConvTranspose2d(ngf,  num_img_channels, 4, 2, 1, bias=False), # 25
            nn.Tanh()
            # nn.LeakyReLU(0.5, inplace=True),
            # 1 x 40 x 12
            # Flatten()
        )
        
    def forward(self, input):
        code = self.encoder(input)
        # code = code.view(code.size(0), code.size(1), 1, 1)
        output = self.decoder(code)
        return output
        
ngpu = 1
netG = InvGenerator(ngpu)
if not args.cpu:
    netG.cuda()

# netG.apply(weights_init)

if args.cp != '':
    netG.load_state_dict(torch.load(args.cp))
    print("Checkpoint '%s' loaded" %(args.cp))
    
# print(netG)
criterion1 = nn.L1Loss()
#optimizerG = optim.SGD(netG.parameters(), lr=0.1, momentum=0.9)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
#exp_lr_scheduler = StepLR(optimizerG, step_size=10, gamma=args.gamma)

def train(netG, train_loader, args):
    for epoch in range(args.epoch):
        disp_loss1 = 0
        disp_loss2 = 0
        disp_iters = 0
        #exp_lr_scheduler.step()
        start = time.time()
        
        for i, data in enumerate(train_loader):
            resp_tensor, truth = data
            if not args.cpu:
                resp_tensor, truth = resp_tensor.cuda(), truth.cuda()

            netG.zero_grad()
            sim = netG(resp_tensor)
            if args.debug:
                pdb.set_trace()
            
            loss_sim1 = criterion1(sim, truth)

            truth2 = truth - pix_offset
            sim1 = sim.detach() - pix_offset
            sim2 = sim1.clone()
            sim2[sim1>=0.5]= 1
            sim2[sim1< 0.5]= 0

            loss_sim2 = criterion1(sim2, truth2)
            loss_sim1.backward()
            optimizerG.step()
            disp_loss1 += loss_sim1.item()
            disp_loss2 += loss_sim2.item()
            disp_iters += 1
            if (i+1) % 10 == 0 or i+1 == len(train_loader):
                disp_loss1 /= disp_iters
                disp_loss2 /= disp_iters
                print('[%d/%d][%d/%d] loss1: %.4f, loss2: %.4f'
                        % (epoch, args.epoch, i+1, len(train_loader), disp_loss1, disp_loss2))

                #calc_tensor_stats(truth.detach(), do_print=True, arr_name='truth')
                #calc_tensor_stats(sim.detach(), do_print=True, arr_name='sim')

                disp_loss1 = 0
                disp_loss2 = 0
                
                disp_iters = 0
                truth2 = truth - pix_offset
                vutils.save_image(truth2, '%s/%d-%02d-real2f.png' %(sim_train_folder, epoch, i),
                                    normalize=False)
                sim2 = sim.detach() - pix_offset
                #simulated_image =sim2
                sim2[sim2>=0.5]= 255
                sim2[sim2<0.5]= 0
                # import pdb; pdb.set_trace()
                vutils.save_image(sim2,   '%s/%d-%02d-simu2f.png' %(sim_train_folder, epoch, i),
                                    normalize=False)
                 
                truth2[truth2==0] = 0
                truth2[truth2==1] = 255
        
                
                vutils.save_image((abs(truth2 - sim2)),   '%s/%d-%02d-diff.png' %(sim_train_folder, epoch, i),
                                    normalize=False)
                
        end = time.time()
        print("Iter %d took %.1f seconds" %(epoch, end-start))
        start = time.time()
        
        if (epoch+1) % 5 == 0:
            # do checkpointing
            cp_path = '%s/gen-%d.pth' % (model_folder, epoch+1)
            torch.save(netG.state_dict(), cp_path)
            print("Saved model checkpoint '%s'" %cp_path)
        
def predict(netG, val_loader, args):
    total_loss1 = 0
    total_loss2 = 0
    disp_loss1 = 0
    disp_loss2 = 0
    
    disp_iters = 0
    
    start = time.time()
    
    for i, data in enumerate(val_loader):
        resp_tensor, truth = data
        if not args.cpu:
            resp_tensor, truth = resp_tensor.cuda(), truth.cuda()

        with torch.no_grad():
            sim = netG(resp_tensor)

        truth2 = truth - pix_offset
        sim1 = sim.detach() - pix_offset
        sim2 = sim1.clone()
        sim2[sim1>=0.5]= 1
        sim2[sim1< 0.5]= 0

        loss_sim1 = criterion1(sim1, truth2)
        loss_sim2 = criterion1(sim2, truth2)
        
        sim2 *= 255
        truth2 *= 255
        
        disp_loss1 += loss_sim1.item()
        total_loss1 += loss_sim1.item()
        disp_loss2 += loss_sim2.item()
        total_loss2 += loss_sim2.item()
        disp_iters += 1
        if (i+1) % 10 == 0 or i+1 == len(val_loader):
            disp_loss1 /= disp_iters
            disp_loss2 /= disp_iters
            end = time.time()
            print('[%d/%d] t %.3f loss1: %.4f, loss2: %.4f' % (i+1, len(val_loader), 
                        end-start, disp_loss1, disp_loss2))
            #calc_tensor_stats(truth.detach(), do_print=True, arr_name='truth')
            #calc_tensor_stats(sim.detach(), do_print=True, arr_name='sim')

            start = time.time()
            disp_loss1 = 0
            disp_loss2 = 0
            disp_iters = 0
            
        vutils.save_image(truth2, '%s/%02d-real.png' %(sim_val_folder, i),
                               normalize=False)
                               
        vutils.save_image(sim1,   '%s/%02d-simu1.png' % (sim_val_folder, i),
                                    normalize=False)
        vutils.save_image(sim2,   '%s/%02d-simu2.png' % (sim_val_folder, i),
                                    normalize=False)
        
        '''
        import sys
        np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
        
        C = torch.ones(1, 1, 40, 12)
        #import pdb
        #pdb.set_trace()
        counter = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        counter.weight.data[...] = 1.0
        neighbor_count = counter(C).data
        energy_thres = 0.5 * neighbor_count
        Ms = torch.zeros(4, 1, 1, 40, 12)
        
        for k in range(3):
            Ms[0] = sim2[k].unsqueeze_(0) #torch.randint(2, size=(1, 1, 40, 12))
            for j in range(3):
                M_energy = counter(Ms[j])
                Ms[j+1][M_energy <  energy_thres] = 0
                Ms[j+1][M_energy > energy_thres] = 1
                # avoid the slight bias towards 1
                Ms[j+1][M_energy == energy_thres] = Ms[j][M_energy == energy_thres]
                
            Ms2 = Ms.squeeze(2)
            vutils.save_image(Ms2, '%d.png' %k, normalize=False)
            ratio_1 = Ms2.sum(dim=3).sum(dim=2).squeeze(1) / (40*12)
            ratio_1 = ratio_1.numpy()
            print("%d: %s" %(k, ratio_1))
        
        '''
            
        diff1 = (((abs(truth2 - sim1)).cpu().numpy()).sum()) / (480*64)
        vutils.save_image((abs(truth2 - sim1)),   '%s/%02d-diff1.png' % (sim_val_folder, i),normalize=False)
        diff2 = (((abs(truth2 - sim2)).cpu().numpy()).sum()) / (480*64)
        vutils.save_image((abs(truth2 - sim2)),   '%s/%02d-diff2.png' % (sim_val_folder, i),normalize=False)
        
        # print (diff)
                                    
        
        
    print( "Total average loss1: %.3f, loss2: %.3f" %(total_loss1 / len(val_loader), total_loss2 / len(val_loader)) )

import time
import datetime
if not args.eval:
    train(netG, train_loader, args)
start = datetime.datetime.now()

predict(netG, val_loader, args)
end = datetime.datetime.now()
print (end-start)
