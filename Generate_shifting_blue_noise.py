#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:47:05 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_shifting_blue_noise(expected_dims):
    
    blue_noise_slice = np.load('blue_noise_cubes/bluenoise1024.npy')
    # blue_noise_cube1 = np.transpose(np.load('blue_noise_cubes/bluenoisecube.npy'), (2,1,0))
    # blue_noise_cube2 = np.transpose(np.load('blue_noise_cubes/bluenoisecube2.npy'), (2,1,0))
    # blue_noise_cube3 = np.transpose(np.load('blue_noise_cubes/bluenoisecube3.npy'), (2,1,0))
    # blue_noise_cube4 = np.transpose(np.load('blue_noise_cubes/bluenoisecube4.npy'), (2,1,0))
    
    # blue_noise_cube5 = np.transpose(np.load('blue_noise_cubes/bluenoisecube5.npy'), (2,1,0))
    # blue_noise_cube6 = np.transpose(np.load('blue_noise_cubes/bluenoisecube6.npy'), (2,1,0))
    # blue_noise_cube7 = np.transpose(np.load('blue_noise_cubes/bluenoisecube7.npy'), (2,1,0))
    # blue_noise_cube8 = np.transpose(np.load('blue_noise_cubes/bluenoisecube8.npy'), (2,1,0))
    
    # concat1=np.concatenate((blue_noise_cube1,blue_noise_cube2),axis=0)
    # concat2=np.concatenate((blue_noise_cube3,blue_noise_cube4),axis=0)
    
    # concat3=np.concatenate((blue_noise_cube5,blue_noise_cube6),axis=0)
    # concat4=np.concatenate((blue_noise_cube7,blue_noise_cube8),axis=0)
    
    # concat5=np.concatenate((concat1,concat2,concat3,concat4),axis=1)
    
    # concat_crop = concat5[0:expected_dims[0],0:expected_dims[1],0:expected_dims[2]]
    
    # blue_noise_slice=concat_crop[:,:,0]
    matrix=blue_noise_slice[0:expected_dims[0],0:expected_dims[1]]
    blue_noise_mask=np.zeros(expected_dims)
    blue_noise_mask[:,:,0]=matrix
    def matrix_slice(matrix,direction,pixels,expected_dims):
        new_slice=np.zeros((expected_dims[0],expected_dims[1]))
        if(direction==1):##down shift
            print('DOWN SHIFT')
            m1=matrix[:expected_dims[0]-pixels,:]
            m2=matrix[expected_dims[0]-pixels:,:]
            print(m2.shape)
            print(m1.shape)
            new_slice[:m2.shape[0],:]=m2
            new_slice[m2.shape[0]:,:]=m1
        if(direction==2):##up shift
            print('UP SHIFT')
            m1=matrix[pixels:,:]
            m2=matrix[:pixels,:]
            print(m1.shape)
            print(m2.shape)
            new_slice[:m1.shape[0],:]=m1
            new_slice[m1.shape[0]:,:]=m2
            
        if(direction==3):##rigth shift
            print('RIGTH SHIFT')
            m1=matrix[:,:expected_dims[1]-pixels]
            m2=matrix[:,expected_dims[1]-pixels:]
            print(m2.shape,m1.shape)
            new_slice[:,:m2.shape[1]]=m2
            new_slice[:,m2.shape[1]:]=m1
        if(direction==4):##left shift
            print('LEFT SHIFT')
            m1=matrix[:,pixels:]
            m2=matrix[:,:pixels]
            print(m1.shape,m2.shape)
            new_slice[:,:m1.shape[1]]=m1
            new_slice[:,m1.shape[1]:]=m2
            
        return new_slice
    
    direction_count=1
    pixel_count=1
    
    for n in range(expected_dims[2]-1):
        new_slice=matrix_slice(matrix=matrix,
                               direction=direction_count,
                               pixels=pixel_count,
                               expected_dims=expected_dims)
        
        direction_count+=1
        if(direction_count>4):
           pixel_count+=1
           direction_count=1
           
        plt.imshow(new_slice,cmap='gray')
        plt.show()
        blue_noise_mask[:,:,n+1]=new_slice
        
    return blue_noise_mask
        
def generate_binary_blue_noise_mask(blue_noise_mask,subsampling_percentage):
    blue_noise_mask=blue_noise_mask/np.max(blue_noise_mask)

    binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
    binary_blue_noise_mask = binary_blue_noise_mask*1
    total = binary_blue_noise_mask.sum()
    
    
    missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
    print('Blue noise missing data: ', missing_data)
    
    return binary_blue_noise_mask.astype(np.uint8)

