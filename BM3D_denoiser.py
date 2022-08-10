#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:42:09 2022

@author: diego
"""
import bm3d
from skimage import img_as_float
import time
import numpy as np

def BM3D_denoiser(volume,sigma_psd=0.1):
    start = time.process_time()
    noisy_img=  img_as_float(volume)
    BM3D_denoised=bm3d.bm3d(noisy_img,
                            sigma_psd,
                            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
    print('TIME ELAPSED FOR DENOISING VOLUME:', time.process_time() - start, 's')
    return (BM3D_denoised*255).astype(np.uint8)
