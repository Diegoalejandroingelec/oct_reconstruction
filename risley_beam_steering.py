#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:42:09 2022

@author: diego
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

def risley_optical_index_fused_silica(l):
    n=np.sqrt((((0.6961663)*l**2)/(l**2-0.0684043**2))+((0.4079426*l**2)/(l**2-0.1162414**2))+((0.8974794*l**2)/(l**2-9.896161**2))+1)
    return n

def get_risley_scan_pattern(PRF,tf,w,phi,n,risley_angle,expected_dims,shift_step):
    #Number Of laser pulses in image capture time
    num_pulse=tf*PRF
    #laser spot number
    i=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
    #Time of laser Pulses
    t1=i*(1/PRF)
    #Angle of risley 1
    angle_risley_1= w*t1*2*np.pi
    #Angle of risley 2
    angle_risley_2=2*np.pi*phi*t1+np.pi
    
    first_deflection_angle=math.asin((1/n)*math.sin(risley_angle))
    
    Bxi=first_deflection_angle*np.cos(angle_risley_1)
    Byi=first_deflection_angle*np.sin(angle_risley_1)
    
    gamma_xi=first_deflection_angle*np.cos(angle_risley_2)
    gamma_yi=first_deflection_angle*np.sin(angle_risley_2)
    
    x=Bxi+gamma_xi
    y=Byi+gamma_yi
    
    
    x_min=np.min(x)
    y_min=np.min(y)
    x_factor=np.abs((expected_dims[1]/2)/x_min)
    y_factor=np.abs((expected_dims[2]/2)/y_min)
    
    x=(x*x_factor)
    y=(y*y_factor)
    
    x = x+np.abs(np.min(x))
    y = y+np.abs(np.min(y))
    # plt.plot(x,y,'ro')
    # plt.show()
    risley_pattern_2D=np.zeros((expected_dims[1],expected_dims[2]))
    for i in range(int(num_pulse)):
        discrete_points_x=round(x[i]) if (round(x[i])>0 and round(x[i])<expected_dims[1]) else expected_dims[1]-1 if round(x[i])>=expected_dims[1] else 0
        discrete_points_y=round(y[i]) if (round(y[i])>0 and round(y[i])<expected_dims[2]) else expected_dims[2]-1 if round(y[i])>=expected_dims[2] else 0
        
        risley_pattern_2D[discrete_points_x,discrete_points_y]=1
       
    
    
    mask=np.zeros((expected_dims[1],expected_dims[2]))
    for shift in range(0,expected_dims[2]+shift_step,shift_step):
        M = np.float32([
        	[1, 0, shift-expected_dims[2]/2],
        	[0, 1, 0]
        ])
        risley_pattern_2D_shifted = cv2.warpAffine(risley_pattern_2D, M, (expected_dims[2],expected_dims[1]))
        mask+=risley_pattern_2D_shifted
        
    mask=(mask!=0)*1

    print('TRANSMTTANCE C-SCAN',(mask.sum()*100)/(expected_dims[1]*expected_dims[2]))
    return mask
def get_risley_3D_mask(expected_dims,
                       PRF,
                       tf,
                       w,
                       phi,
                       risley_angle,
                       shift_step,
                       band_width,
                       line_width,
                       start_wavelength):
    

    
    mask_risley=np.zeros(expected_dims)
    for i in range(1,expected_dims[0]+1):
        # print(i*line_width+start_wavelength)
        
        #Risley optical index fused silica
        n=risley_optical_index_fused_silica(i*line_width+start_wavelength)
        #print('Risley optical index fused silica ',n)
        mask_2D=get_risley_scan_pattern(PRF+(i*(512/50)),
                                tf,
                                w,
                                phi,
                                n,
                                risley_angle,
                                expected_dims,
                                shift_step)
        
        mask_risley[i-1,:,:]=mask_2D
        
    # for m in range(512):   
    #     cv2.imshow('Risley_pattern_2D_shifted_C_Scan',mask_risley[m,:,:].astype(np.uint8)*255)
        
    #     # cv2.imshow('Risley_pattern_2D_shifted_B_Scan',mask_risley[:,:,0].astype(np.uint8)*255)
    #     cv2.waitKey(500)
    #     cv2.destroyAllWindows()
    for m in range(100):
        print('TRANSMTTANCE B-SCAN',(mask_risley[:,:,m].sum()*100)/(expected_dims[0]*expected_dims[1]))
    total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
    print('TOTAL TRANSMITTANCE: ',total_transmittance)
    return mask_risley.astype(np.uint8)

expected_dims=(512,1000,100) 
#Laser Pulse Rate
PRF=199900
#Image Capture Time 0.003
tf=0.03

#angular speed risley 1 rotations per sec
w=400
#angula speed risley 2 rotations per sec
phi=w/0.09

risley_angle=1*(np.pi/180)

shift_step=17
band_width=176
line_width=band_width/expected_dims[0]
start_wavelength=962

mask_risley=get_risley_3D_mask(expected_dims,
                        PRF,
                        tf,
                        w,
                        phi,
                        risley_angle,
                        shift_step,
                        band_width,
                        line_width,
                        start_wavelength)



# for n in range(100):
#     cv2.imshow('Risley_pattern_2D',mask_risley[:,:,n].astype(np.uint8)*255)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume  
  
volume = read_data('/home/diego/Documents/Delaware/tensorflow/training_3D_images/oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1023.mat')

plt.imshow(volume[:,:,50], cmap='gray')
plt.show()

sub=np.multiply(mask_risley,volume).astype(np.uint8)
for i in range(100):
    plt.imshow(volume[:,:,i], cmap='gray')
    plt.show()
    
for i in range(100):
    plt.imshow(sub[:,:,i], cmap='gray')
    plt.show()
    
    
    
    
    