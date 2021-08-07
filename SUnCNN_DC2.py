# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:53:50 2020

@author: behnood
"""

from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from utils.denoising_utils import *

from skimage._shared import *
from skimage.util import *
from skimage.metrics.simple_metrics import _as_floats
from skimage.metrics.simple_metrics import mean_squared_error


from UtilityMine import *
from utils.sr_utils import tv_loss
from numpy import linalg as LA

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = False

import scipy.io
#%%
#%%
fname2  = "Data/DC2/Y_clean.mat"
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Y_clean"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
#%%
fname3  = "Data/DC2/XT.mat"
mat3 = scipy.io.loadmat(fname3)
A_true_np = mat3["XT"]

#%%
fname4  = "Data/DC2/EE.mat"
mat4 = scipy.io.loadmat(fname4)
EE = mat4["EE"]
#%%
LibS=EE.shape[1]
#%%
npar=np.zeros((1,3))
npar=np.zeros((1,3))
npar[0,0]=14.7
npar[0,1]=46.5
npar[0,2]=147
#npar[0,3]=367
tol1=npar.shape[1]
tol2=1
save_result=False
import time
from tqdm import tqdm
rmax=9
for fi in tqdm(range(tol1)):
    for fj in tqdm(range(tol2)):
            #%%
        img_np_gt=np.clip(img_np_gt, 0, 1)
        img_noisy_np = add_noise(img_np_gt, 1/npar[0,fi])#11.55 20 dB, 36.7 30 dB, 116.5 40 dB
        print(compare_snr(img_np_gt, img_noisy_np))
        img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC=np.diag(SS)@U
        img_resh_DN=V[:,:rmax]@PC[:rmax,:]
        img_noisy_np=np.reshape(np.clip(img_resh_DN, 0, 1),(p1,nr1,nc1))
        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        need_bias=True
        OPT_OVER = 'net' # 'net,input'
        
        # 
        reg_noise_std = 0.0
        LR1 = 0.001
        
        OPTIMIZER1='adam'# 'RMSprop'#'adam' # 'LBFGS'
        show_every = 500
        exp_weight=0.99
        if fi==0:
            num_iter1 = 4000
        elif fi==1:
            num_iter1 = 8000
        elif fi==2:
            num_iter1 = 12000
        input_depth =img_noisy_np.shape[0]
        class CAE_AbEst(nn.Module):
            def __init__(self):
                super(CAE_AbEst, self).__init__()
                self.conv1 = nn.Sequential(
                    UnmixArch(
                            input_depth, EE.shape[1],
                            # num_channels_down = [8, 16, 32, 64, 128], 
                            # num_channels_up   = [8, 16, 32, 64, 128],
                            # num_channels_skip = [4, 4, 4, 4, 4], 
                            num_channels_down = [ 256],
                            num_channels_up =   [ 256],
                            num_channels_skip =    [ 4],  
                            filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                            upsample_mode='bilinear', # downsample_mode='avg',
                            need1x1_up=True,
                            need_sigmoid=True, need_bias=True, pad=pad, act_fun='ReLU').type(dtype)
                )
        
            def forward(self, x):
                x = self.conv1(x)
                return x

        net1 = CAE_AbEst()
        net1.cuda()
        print(net1)

        # Compute number of parameters
        s  = sum([np.prod(list(p11.size())) for p11 in net1.parameters()]); 
        print ('Number of params: %d' % s)
        
        # Loss
        mse = torch.nn.MSELoss().type(dtype)
        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
        # if fk==0:
        net_input1 = get_noise(input_depth, INPUT,
          (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
        # net_input1 = img_noisy_torch 
        E_torch = np_to_torch(EE).type(dtype)
        #%%
        net_input_saved = net_input1.detach().clone()
        noise = net_input1.detach().clone()
        out_avg = None
        out_HR_avg= None
        last_net = None
        RMSE_LR_last = 0
        loss=np.zeros((num_iter1,1))
        AE=np.zeros((num_iter1,1))
        i = 0
        def closure1():
            
            global i, RMSE_LR, RMSE_LR_ave, RMSE_HR, out_LR_np, out_avg_np, out_LR\
                , out_avg,out_HR_np, out_HR_avg, out_HR_avg_np, RMSE_LR_last, last_net\
                    , net_input,RMSE_LR_avg,RMSE_HR_avg,RE_HR_avg, RE_HR, Eest,loss,AE\
                       , MAE_LR,MAE_LR_avg,MAE_HR,MAE_HR_avg
            
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            
            out_LR = net1(net_input1)
            out_HR=torch.mm(E_torch.view(p1,LibS),out_LR.view(LibS,nr1*nc1))
            # Smoothing
            if out_avg is None:
                out_avg = out_LR.detach()
                out_HR_avg = out_HR.detach()
            else:
                out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)
                out_HR_avg = out_HR_avg * exp_weight + out_HR.detach() * (1 - exp_weight)

        #%%
            out_HR=out_HR.view((1,p1,nr1,nc1))
            total_loss = mse(img_noisy_torch, out_HR)
            total_loss.backward()
            if 1:
             out_LR_np = out_LR.detach().cpu().squeeze().numpy()
             out_avg_np = out_avg.detach().cpu().squeeze().numpy()
             SRE=10*np.log10(LA.norm(A_true_np.astype(np.float32).reshape((EE.shape[1],nr1*nc1)),'fro')/LA.norm((A_true_np.astype(np.float32)- np.clip(out_LR_np, 0, 1)).reshape((EE.shape[1],nr1*nc1)),'fro'))
             SRE_avg=10*np.log10(LA.norm(A_true_np.astype(np.float32).reshape((EE.shape[1],nr1*nc1)),'fro')/LA.norm((A_true_np.astype(np.float32)- np.clip(out_avg_np, 0, 1)).reshape((EE.shape[1],nr1*nc1)),'fro'))
             MAE_LR= 100*np.mean(abs(A_true_np.astype(np.float32)- np.clip(out_LR_np, 0, 1)))
             MAE_LR_avg= 100*np.mean(abs(A_true_np.astype(np.float32)- np.clip(out_avg_np, 0, 1)))
             print ('Iteration %05d    Loss %f   MAE_LR: %f MAE_LR_avg: %f  SRE: %f SRE_avg: %f' % (i, total_loss.item(), MAE_LR, MAE_LR_avg, SRE, SRE_avg), '\r', end='')

            if  PLOT and i % show_every == 0:
                out_LR_np = torch_to_np(out_LR)
                out_avg_np = torch_to_np(out_avg)
        #        plot_image_grid([np.clip(out_np, 0, 1), 
        #                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
                
                # out_LR_np = np.clip(out_LR_np, 0, 1)
                # out_avg_np = np.clip(out_avg_np, 0, 1)
                
                # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,10))
                # ax1.imshow(np.stack((out_LR_np[2,:,:],out_LR_np[1,:,:],out_LR_np[0,:,:]),2))
                # ax2.imshow(np.stack((out_avg_np[2,:,:],out_avg_np[1,:,:],out_avg_np[0,:,:]),2))
                # ax3.imshow(np.stack((A_true_np[2,:,:],A_true_np[1,:,:],A_true_np[0,:,:]),2))
                # plt.show()
                plt.plot(out_LR_np.reshape(LibS,nr1*nc1))
            loss[i]=total_loss.item() 
            i += 1
        
            return total_loss
        
        p11 = get_params(OPT_OVER, net1, net_input1)
        optimize(OPTIMIZER1, p11, closure1, LR1, num_iter1)
        if 1:
            out_LR_np = out_LR.detach().cpu().squeeze().numpy()
            out_avg_np = out_avg.detach().cpu().squeeze().numpy()
            MAE_LR_avg= 100*np.mean(abs(A_true_np.astype(np.float32)- np.clip(out_avg_np, 0, 1)))
            MAE_LR= 100*np.mean(abs(A_true_np.astype(np.float32)- np.clip(out_LR_np, 0, 1)))
            SRE=10*np.log10(LA.norm(A_true_np.astype(np.float32).reshape((EE.shape[1],nr1*nc1)),'fro')/LA.norm((A_true_np.astype(np.float32)- np.clip(out_LR_np, 0, 1)).reshape((EE.shape[1],nr1*nc1)),'fro'))
            SRE_avg=10*np.log10(LA.norm(A_true_np.astype(np.float32).reshape((EE.shape[1],nr1*nc1)),'fro')/LA.norm((A_true_np.astype(np.float32)- np.clip(out_avg_np, 0, 1)).reshape((EE.shape[1],nr1*nc1)),'fro'))
            print ('Iteration %05d  MAE_LR: %f MAE_LR_avg: %f  SRE: %f SRE_avg: %f ' % (i, MAE_LR, MAE_LR_avg, SRE, SRE_avg), '\r', end='')
        # if  save_result is True:
        #      scipy.io.savemat("C:/Users/behnood/Desktop/Sparse Unmixing/Results/Sim2/demo1/10runs/out_avg_np%01d%01d.mat" % (fi+2,fj+1),
        #                     {'out_avg_np%01d%01d' % (fi+2, fj+1):out_avg_np.transpose(1,2,0)})
        #      scipy.io.savemat("C:/Users/behnood/Desktop/Sparse Unmixing/Results/Sim2/demo1/10runs/out_LR_np%01d%01d.mat" % (fi+2,fj+1),
        #                     {'out_LR_np%01d%01d' % (fi+2, fj+1):out_LR_np.transpose(1,2,0)})
#%%
