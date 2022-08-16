#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:46:52 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
import random

vol_dims=(512,1000,100)


def gauss_map(size_x, size_y=None, sigma_x=5, sigma_y=None, center=None):    
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x
    
    assert isinstance(size_x, int)
    assert isinstance(size_y, int)
    
    if center is None:
        x0 = size_x // 2
        y0 = size_y // 2
    else:
        x0 = center[0]
        y0 = center[1]

    
    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:,np.newaxis]
    
    x -= x0
    y -= y0
    
    exp_part = x**2/(2*sigma_x**2)+ y**2/(2*sigma_y**2)
    return 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-exp_part)

#gaussian2D=gauss_map(512,100, sigma_x=90, sigma_y=35, center=(200,50))


#gaussian2D=(gaussian2D-np.min(gaussian2D))/(np.max(gaussian2D)-np.min(gaussian2D))


#plt.imshow(gaussian2D)

def generate_gaussian_blue_noise_mask(original_volume_path,desired_transmittance,sigma,plot_mask):
    def gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def load_obj(name):
        with open( name, 'rb') as f:
            return pickle.load(f)
        
    def create_blue_noise_mask_1(expected_dims,subsampling_percentage):
        blue_noise_cube1 = np.transpose(np.load('blue_noise_cubes/bluenoisecube.npy'), (2,1,0))
        blue_noise_cube2 = np.transpose(np.load('blue_noise_cubes/bluenoisecube2.npy'), (2,1,0))
        blue_noise_cube3 = np.transpose(np.load('blue_noise_cubes/bluenoisecube3.npy'), (2,1,0))
        blue_noise_cube4 = np.transpose(np.load('blue_noise_cubes/bluenoisecube4.npy'), (2,1,0))
        
        blue_noise_cube5 = np.transpose(np.load('blue_noise_cubes/bluenoisecube5.npy'), (2,1,0))
        blue_noise_cube6 = np.transpose(np.load('blue_noise_cubes/bluenoisecube6.npy'), (2,1,0))
        blue_noise_cube7 = np.transpose(np.load('blue_noise_cubes/bluenoisecube7.npy'), (2,1,0))
        blue_noise_cube8 = np.transpose(np.load('blue_noise_cubes/bluenoisecube8.npy'), (2,1,0))
        
        concat1=np.concatenate((blue_noise_cube1,blue_noise_cube2),axis=0)
        concat2=np.concatenate((blue_noise_cube3,blue_noise_cube4),axis=0)
        
        concat3=np.concatenate((blue_noise_cube5,blue_noise_cube6),axis=0)
        concat4=np.concatenate((blue_noise_cube7,blue_noise_cube8),axis=0)
        
        concat5=np.concatenate((concat1,concat2,concat3,concat4),axis=1)
        
        blue_noise_mask = concat5[0:expected_dims[0],0:expected_dims[1],0:expected_dims[2]]
        
        blue_noise_mask=blue_noise_mask/np.max(blue_noise_mask)
        
        binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
        binary_blue_noise_mask = binary_blue_noise_mask*1
        total = binary_blue_noise_mask.sum()
        
        
        missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
        print('Blue noise missing data: ', missing_data)
        
        return blue_noise_mask,binary_blue_noise_mask
        
     
    original_volume =load_obj(original_volume_path)


    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)

    means=np.argmax(mean_b_scans,0)
    #means_smooth=savgol_filter(means,51,1)


    # plt.imshow(mean_b_scans,cmap='gray')
    # plt.plot(means_smooth)
    # plt.show()


    gaussian_mask=np.ones((512,1000))

    for i in range(vol_dims[1]):
        for j in range(vol_dims[0]):
            likelihood=gaussian(j, 1, means[i], sigma)
            threshold= random.uniform(0, 1)
            if(threshold>likelihood):
               gaussian_mask[j,i]=0
        
      
    gaussian_mask = np.repeat(gaussian_mask[None,:], vol_dims[2], axis=0)
    gaussian_mask=np.transpose(gaussian_mask,(1,2,0))


    # plt.imshow(gaussian_mask[:,:,11],cmap='gray')
    # plt.show() 


    gaussian_mask_transmittance=(gaussian_mask.sum())/(vol_dims[0]*vol_dims[1]*vol_dims[2])
    # print(gaussian_mask_transmittance)


    desired_transmittance=0.25

    blue_noise_transmitance=desired_transmittance/gaussian_mask_transmittance

    _,blue_noise_mask=create_blue_noise_mask_1(vol_dims,subsampling_percentage=1-blue_noise_transmitance)



    mask = np.multiply(gaussian_mask.astype(np.uint8),blue_noise_mask.astype(np.uint8))

    print('Missing Data: ', 100-(mask.sum()*100)/(vol_dims[0]*vol_dims[1]*vol_dims[2]))
    if(plot_mask):
        plt.imshow(mask[:,:,11],cmap='gray')
        plt.show() 
    return mask,original_volume

original_volume_path ='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_volumes/Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl'

import time
start = time.process_time()
mask, original_volume=generate_gaussian_blue_noise_mask(original_volume_path,desired_transmittance=0.25,sigma=51,plot_mask=True) 
print(time.process_time() - start)


sub_sampled_volume=np.multiply(mask,original_volume)



























    