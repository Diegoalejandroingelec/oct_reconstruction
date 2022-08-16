#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 22:59:48 2022

@author: diego
"""

from scipy.io import loadmat,savemat
from glob import glob
import random
import matplotlib.pyplot as plt
import os
from BM3D_denoiser import median_filter_3D

def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume


def get_volume_paths():
    amd_eyes_paths = glob("../oct_original_volumes/AMD/*.mat", recursive = True)
    control_eyes_paths= glob("../oct_original_volumes/Control/*.mat", recursive = True)
    all_paths = amd_eyes_paths+control_eyes_paths
    random.shuffle(all_paths)
    return all_paths



dataset_folder='DATASET_DENOISED'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
    
    
all_paths=get_volume_paths()
 
for volume_path in all_paths:
    name=volume_path.split('/')[-1]
    volume_path=volume_path.strip('\n')
    print(volume_path) 
    volume = read_data(volume_path)
    denoised_volume=median_filter_3D(volume,40,5)
    mdic = {"images": denoised_volume}
    savemat('./'+dataset_folder+f"/denoised_{name}", mdic)
    
# volume1 = read_data('./DATASET_DENOISED/denoised_Farsiu_Ophthalmology_2013_AMD_Subject_1231.mat')
# plt.imshow(volume1[:,:,0],cmap='gray')
# plt.show()