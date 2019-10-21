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
from ranger import Ranger
from mish import Mish
from torch.utils.tensorboard import SummaryWriter
board = SummaryWriter()

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
    parser.add_argument('--lr', type=float, default=-1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.4, help='gamma for learning rate, default=0.4')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--debug', action='store_true', help='enable debug')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--cp', default='', help="path to checkpoint (to continue training or evaluation)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--eval', help='Only do evaluation', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--opt', dest='optimizer',
                      help='The type of optimizer',
                      default='ranger', type=str)
    parser.add_argument('--act', dest='act_func',
                      help='Activation functions in the encoder/decoder',
                      default='mish,mish', type=str)
                      
    parser.add_argument('--nomix', dest='use_mixconv',
                      help='Do not use mixture of convs',
                      action='store_false')                      
    parser.add_argument('--mix', dest='num_mix_comps',
                        help='Number of mixture conv components',
                        default=10, type=int)
    parser.add_argument('--nodec', dest='use_decoder',
                      help='Do not use decoder (directly output image)',
                      action='store_false')
                      
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

args = parse_args()
print(args)

if not args.use_mixconv:
    args.num_mix_comps = 1
if args.use_decoder:
    folder_prefix = 'dec%02d' %args.num_mix_comps
else:
    folder_prefix = 'nodec'
print("Setting: %s" %folder_prefix)

train_out_folder = folder_prefix + '-train'
val_out_folder   = folder_prefix + '-val'
model_folder     = folder_prefix + '-model'

for path in (train_out_folder, val_out_folder, model_folder):
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
                    "/home/shaohua/ndt/conv_dataset/real_groundtruth1", 
                    "/home/shaohua/ndt/conv_dataset/real_groundtruth3", 
                    "/home/shaohua/ndt/conv_dataset/real_groundtruth6", 
                    "/home/shaohua/ndt/conv_dataset/imag_groundtruth1",
                    "/home/shaohua/ndt/conv_dataset/imag_groundtruth3",
                    "/home/shaohua/ndt/conv_dataset/imag_groundtruth6"
                 ]
                      
ndt_trainset = NDTDataset(resp_folders, (0, 16000),
                         "/home/shaohua/ndt/conv_dataset/data12f2",
                          pix_offset, transforms.ToTensor(), 0.0)

ndt_validset = NDTDataset(resp_folders, (16000, 20000),
                          "/home/shaohua/ndt/conv_dataset/data12f2",
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

nc = len(resp_folders)                                       

class Flatten(nn.Module):
    def __init__(self, h, w):
        super(Flatten, self).__init__()
        self.h = h
        self.w = w
        
    def forward(self, input):
        flat_feat = input.view(input.size(0), -1, 1, 1)
        output = flat_feat.reshape(flat_feat.size(0), 1, self.h, self.w)
        return output

class Power(nn.Module):
    def __init__(self, p):
        super(Power, self).__init__()
        self.p = p
        
    def forward(self, input):
        return torch.pow(input, self.p)

class InvGenerator(nn.Module):
    def __init__(self, ngpu, use_decoder, use_mixconv, act_func):
        super(InvGenerator, self).__init__()
        self.ngpu = ngpu
        self.use_decoder = use_decoder
        self.use_mixconv = use_mixconv
        
        self.enc_act_func, self.dec_act_func = act_func
        if self.enc_act_func == 'mish':
            EncActFuncs = [ Mish() for i in range(4) ]
        elif self.enc_act_func == 'relu':
            EncActFuncs = [ nn.LeakyReLU(0.3, inplace=True) for i in range(4) ]
            
        if self.dec_act_func == 'mish':
            DecActFuncs = [ Mish() for i in range(4) ]
        elif self.dec_act_func == 'relu':
            DecActFuncs = [ nn.ReLU(inplace=True) for i in range(4) ]
        
        C1 = C2 = C3 = C4 = 5
        encoder = [
                    # input is (nc=1) x 40 x 40
                    nn.Conv2d(nc, ndf * C1, 6, 1, 0, bias=False),
                    nn.BatchNorm2d(ndf * C1),
                    EncActFuncs[0],
                    # state size. (ndf * 2) x 35 x 35
                    nn.Conv2d(ndf * C1, ndf * C2, 5, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * C2),
                    EncActFuncs[1],
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * C2, ndf * C3, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * C3),
                    EncActFuncs[2],
                    # state size. (ndf*2) x 8 x 8
                    nn.Conv2d(ndf * C3, ndf * C4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * C4),
                    EncActFuncs[3],
                    # state size. (ndf*4) x 4 x 4
                    # nn.Conv2d(ndf * C4, nz, 4, 1, 0, bias=False),
                    # nn.BatchNorm2d(ndf * 4),
                    # nn.ReLU(True),
                    # state size. (ndf*8) x 3 x 3
                    # nn.Conv2d(ndf * 4, nz, 3, 1, 0, bias=False),
                    # 480 * 1 * 1
                  ]
        
        if self.use_decoder:
            encoder.append( nn.Conv2d(ndf * C4, nz, 4, 1, 0, bias=False) )
        else:
            encoder.append( nn.Conv2d(ndf * C4, 40 * 12, 4, 1, 0, bias=False) )
            encoder.append(Flatten(40, 12))
            
        self.encoder = nn.Sequential(*encoder)
        
        decoder = [
                    nn.ConvTranspose2d( nz, ngf * 8, 3, 1, 0, bias=False),          # 13
                    nn.BatchNorm2d(ngf * 8),
                    DecActFuncs[0],
                    # image size: (ngf*8) x 3 x 3
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, (2,1), 1, bias=False),  # 16
                    nn.BatchNorm2d(ngf * 4),
                    DecActFuncs[1],
                    # image size: (ngf*4) x 5 x 3
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),      # 19
                    nn.BatchNorm2d(ngf * 2),
                    DecActFuncs[2],
                    # image size: (ngf*2) x 10 x 6
                    nn.ConvTranspose2d(ngf * 2, ngf, (4,3), (2,1), 1, bias=False),  # 22
                    Power(2),
                    nn.BatchNorm2d(ngf),
                    DecActFuncs[3],
                  ]
                  
        if self.use_mixconv:
            # image size: (ngf*2) x 20 x 5
            decoder.append(nn.ConvTranspose2d(ngf,  args.num_mix_comps, 4, 2, 1, bias=False)) # 25
        else:
            decoder.append(nn.ConvTranspose2d(ngf,  num_img_channels, 4, 2, 1, bias=False)) # 25
            decoder.append( nn.Tanh() )
            # 1 x 40 x 12
            # Flatten()
        
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, input):
        code = self.encoder(input)
        if self.use_decoder:
            feats = self.decoder(code)
            if self.use_mixconv:
                weights = torch.softmax(feats, dim=1)
                feat = (weights * feats).sum(dim=1, keepdim=True)
                output = feat
            else:
                output = feats / 1.5
        else:
            output = code
            
        return output
        
ngpu = 1
args.act_func = args.act_func.split(',')
net = InvGenerator(ngpu, args.use_decoder, args.use_mixconv, args.act_func)
if not args.cpu:
    net.cuda()

if args.cp != '':
    net.load_state_dict(torch.load(args.cp))
    print("Checkpoint '%s' loaded" %(args.cp))
    
# print(net)
criterion1 = nn.L1Loss()

param_to_moduleName = {}
for m in net.modules():
    for p in m.parameters(recurse=False):
        param_to_moduleName[p] = type(m).__name__

optimized_params = list( param for param in net.named_parameters() if param[1].requires_grad )
no_decay    = ['bias']
bn_no_decay = ['bias', 'weight']
no_decay_params = []
no_decay_names = []
decay_params = []
for n, p in optimized_params:
    if any(nd in n for nd in no_decay) or \
      (param_to_moduleName[p] == 'BatchNorm2d' and any(nd in n for nd in bn_no_decay)):
        no_decay_params.append(p)
        no_decay_names.append(n)
    else:
        decay_params.append(p)
print(no_decay_names)
weight_decay = 1e-6

grouped_params = [
    { 'params': decay_params, 'weight_decay': weight_decay },
    { 'params': no_decay_params, 'weight_decay': 0.0 }
    ]

default_lr = { 'sgd': 0.01, 'ranger': 0.0002, 'adam': 0.0001 }
if args.lr == -1:
    args.lr = default_lr[args.optimizer]
                                
sgd_optimizer    = optim.SGD(grouped_params, lr=args.lr, momentum=0.9)
exp_lr_scheduler = StepLR(sgd_optimizer, step_size=10, gamma=args.gamma)
adam_optimizer   = optim.Adam(grouped_params, lr=args.lr)
ranger_optimizer = Ranger(grouped_params, lr=args.lr)
if args.optimizer == 'adam':
    optimizer = adam_optimizer
elif args.optimizer == 'ranger':
    optimizer = ranger_optimizer
elif args.optimizer == 'sgd':
    optimizer = sgd_optimizer
    
def train(net, train_loader, args):
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

            optimizer.zero_grad()
            sim = net(resp_tensor)
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
            optimizer.step()
            
            disp_loss1 += loss_sim1.item()
            disp_loss2 += loss_sim2.item()
            disp_iters += 1
            if (i+1) % 10 == 0 or i+1 == len(train_loader):
                disp_loss1 /= disp_iters
                disp_loss2 /= disp_iters
                print('[%d/%d][%03d/%d] loss1: %.4f, loss2: %.4f'
                        % (epoch, args.epoch, i+1, len(train_loader), disp_loss1, disp_loss2))

                #calc_tensor_stats(truth.detach(), do_print=True, arr_name='truth')
                #calc_tensor_stats(sim.detach(), do_print=True, arr_name='sim')

                board_loss1 = disp_loss1 * 100
                board_loss2 = disp_loss2 * 100
                
                disp_loss1 = 0
                disp_loss2 = 0
                
                disp_iters = 0
                truth2 = truth - pix_offset
                vutils.save_image(truth2, '%s/%d-%02d-real2f.png' %(train_out_folder, epoch, i),
                                    normalize=False)
                sim2 = sim.detach() - pix_offset
                #simulated_image =sim2
                sim2[sim2>=0.5]= 255
                sim2[sim2<0.5]= 0
                if args.debug:
                    pdb.set_trace()

                vutils.save_image(sim2,   '%s/%d-%02d-simu2f.png' %(train_out_folder, epoch, i),
                                    normalize=False)
                 
                truth2[truth2==0] = 0
                truth2[truth2==1] = 255
        
                
                vutils.save_image((abs(truth2 - sim2)),   '%s/%d-%02d-diff.png' %(train_out_folder, epoch, i),
                                    normalize=False)
                
        end = time.time()
        print("Epoch %d took %.1f seconds" %(epoch, end-start))
        start = time.time()

        board.add_scalar('loss1', board_loss1, epoch+1)
        board.add_scalar('loss2', board_loss2, epoch+1)
        
        if (epoch+1) % 5 == 0:
            # do checkpointing
            cp_path = '%s/gen-%d.pth' % (model_folder, epoch+1)
            torch.save(net.state_dict(), cp_path)
            print("Saved model checkpoint '%s'" %cp_path)

        if args.optimizer == 'sgd':
            exp_lr_scheduler.step()
                
def predict(net, val_loader, args):
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
            sim = net(resp_tensor)

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
            
        vutils.save_image(truth2, '%s/%02d-real.png' %(val_out_folder, i),
                               normalize=False)
                               
        vutils.save_image(sim1,   '%s/%02d-simu1.png' % (val_out_folder, i),
                                    normalize=False)
        vutils.save_image(sim2,   '%s/%02d-simu2.png' % (val_out_folder, i),
                                    normalize=False)
        
        diff1 = (((abs(truth2 - sim1)).cpu().numpy()).sum()) / (480*64)
        vutils.save_image((abs(truth2 - sim1)),   '%s/%02d-diff1.png' % (val_out_folder, i),normalize=False)
        diff2 = (((abs(truth2 - sim2)).cpu().numpy()).sum()) / (480*64)
        vutils.save_image((abs(truth2 - sim2)),   '%s/%02d-diff2.png' % (val_out_folder, i),normalize=False)
        
        # print (diff)
        
    print( "Total average loss1: %.3f, loss2: %.3f" %(total_loss1 / len(val_loader), total_loss2 / len(val_loader)) )

        
import time
import datetime
if not args.eval:
    train(net, train_loader, args)
start = datetime.datetime.now()

predict(net, val_loader, args)
end = datetime.datetime.now()
print (end-start)
