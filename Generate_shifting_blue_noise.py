#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:47:05 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
expected_dims=(512,512,100)
def generate_shifting_blue_noise_all_directions(expected_dims):
    
    blue_noise_slice = np.load('blue_noise_cubes/bluenoise1024.npy')
    blue_shape=blue_noise_slice.shape
    i_x_start=int(blue_shape[0]/2-expected_dims[0]/2)
    i_x_end=int(blue_shape[0]/2+expected_dims[0]/2)
    
    i_x_start=int(blue_shape[0]/2-expected_dims[0]/2)
    i_x_end=int(blue_shape[0]/2+expected_dims[0]/2)
    
    i_y_start=int(blue_shape[1]/2-expected_dims[1]/2)
    i_y_end=int(blue_shape[1]/2+expected_dims[1]/2)
    
    matrix_i=blue_noise_slice[i_x_start:i_x_end,
                            i_y_start:i_y_end]
    blue_noise_mask=np.zeros(expected_dims)
    blue_noise_mask[:,:,0]=matrix_i
    def matrix_shift_sides(direction,pixels):          
        if(direction==1):##rigth shift
            print('DOWN SHIFT')
            return blue_noise_slice[i_x_start-pixels:i_x_end-pixels,i_y_start:i_y_end]
        if(direction==2):##left shift
            print('UP SHIFT')
            return blue_noise_slice[i_x_start+pixels:i_x_end+pixels,i_y_start:i_y_end]
        if(direction==3):##rigth shift
            print('RIGTH SHIFT')
            if(i_y_end+pixels<=blue_shape[0]):
                return blue_noise_slice[i_x_start:i_x_end,i_y_start+pixels:i_y_end+pixels]
            else:
                px=(i_y_end+pixels)-blue_shape[0]
                matrix=blue_noise_slice[i_x_start:i_x_end,blue_shape[0]-expected_dims[1]:blue_shape[0]]
                m1=matrix[:,:expected_dims[1]-px]
                m2=matrix[:,expected_dims[1]-px:]
                new_slice[:,:m2.shape[1]]=m2
                new_slice[:,m2.shape[1]:]=m1
                #return new_slice
                
        if(direction==4):##left shift
            print('LEFT SHIFT')
            
            if(i_y_start-pixels>=0):
                return blue_noise_slice[i_x_start:i_x_end,i_y_start-pixels:i_y_end-pixels]
                
            else:
                px=(pixels-i_y_start)
                matrix=blue_noise_slice[i_x_start:i_x_end,0:expected_dims[1]]
                m1=matrix[:,px:]
                m2=matrix[:,:px]
                new_slice[:,:m1.shape[1]]=m1
                new_slice[:,m1.shape[1]:]=m2
                #return new_slice
    direction_count=1
    pixel_count=1
    
    for n in range(expected_dims[2]-1):
        new_slice=matrix_shift_sides(direction=direction_count,
                                     pixels=pixel_count)
        
        direction_count+=1
        if(direction_count>2):
            pixel_count+=1
            direction_count=1
           
        # plt.imshow(new_slice,cmap='gray')
        # plt.show()
        blue_noise_mask[:,:,n+1]=new_slice
        
    return blue_noise_mask

def generate_shifting_blue_noise(expected_dims):
    
    blue_noise_slice = np.load('blue_noise_cubes/bluenoise1024.npy')
    # matrix=blue_noise_slice[0:expected_dims[0],0:expected_dims[1]]
    blue_noise_mask=np.zeros(expected_dims)
    #blue_noise_mask[:,:,0]=matrix
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
    
    # direction_count=1
    # pixel_count=1
    
    for n in range(expected_dims[2]):
        # new_slice=matrix_slice(matrix=matrix,
        #                        direction=direction_count,
        #                        pixels=pixel_count,
        #                        expected_dims=expected_dims)
        
        # direction_count+=1
        # if(direction_count>4):
        #    pixel_count+=1
        #    direction_count=1
           

        new_slice=blue_noise_slice[n:expected_dims[0]+n,0:expected_dims[1]]
        # plt.imshow(new_slice,cmap='gray')
        # plt.show()
        blue_noise_mask[:,:,n]=new_slice
        
    return blue_noise_mask
        
def generate_binary_blue_noise_mask(blue_noise_mask,subsampling_percentage):
    blue_noise_mask=blue_noise_mask/np.max(blue_noise_mask)

    binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
    binary_blue_noise_mask = binary_blue_noise_mask*1
    total = binary_blue_noise_mask.sum()
    
    
    missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
    print('Blue noise missing data: ', missing_data)
    
    return binary_blue_noise_mask.astype(np.uint8)

# blue_noise=generate_shifting_blue_noise_all_directions(expected_dims)

# import napari

# viewer = napari.view_image(blue_noise*255)