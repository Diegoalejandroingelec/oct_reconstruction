#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:47:47 2022

@author: diego
"""
import torch
import numpy as np
import torch.nn as nn

def compute_PSNR(original,reconstruction,bit_representation=8):
    mse = nn.MSELoss()
    x=torch.from_numpy(original.astype('float'))
    y=torch.from_numpy(reconstruction.astype('float'))    
    MSE=mse(x, y)
    MSE=MSE.item()
    MAXI=np.power(2,bit_representation)-1
    PSNR=20*np.log10(MAXI)-10*np.log10(MSE)
    return PSNR