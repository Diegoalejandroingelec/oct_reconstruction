#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:17:49 2022

@author: diego
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat

def get_movement():
    x_factor=180
    y_factor=90
    z_factor=140
    fs=350
    
    
    df = pd.read_csv ('pupil-0.csv',header=None)
    df=df.rename(columns={0: 'frame', 1: 'x', 2: 'y', 3: 'z'})
    
    
    time1=np.linspace(0,len(df['x'])-1,len(df['x']))*(1/fs)
    
    
    
    # plt.plot(time1,np.array(df['x'])*1000,label='x')
    # plt.plot(time1,np.array(df['y'])*1000,label='y')
    # plt.plot(time1,np.array(df['z'])*1000,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('mm')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    fx = interp1d(time1,np.array(df['x']))
    fy = interp1d(time1,np.array(df['y'])) 
    fz = interp1d(time1,np.array(df['z'])) 
           
    new_t=np.linspace(0,(140976)-1,140976)/13671.875000000002
                 
    x_interpolation= fx(new_t)
    y_interpolation= fy(new_t)
    z_interpolation= fz(new_t)
    
    # plt.plot(new_t,x_interpolation,'o')
    # plt.plot(time1,np.array(df['x']))
    # plt.show()
    
    # plt.plot(new_t,y_interpolation,'o')
    # plt.plot(time1,np.array(df['y']))
    # plt.show()
    
    # plt.plot(new_t,z_interpolation,'o')
    # plt.plot(time1,np.array(df['z']))
    # plt.show()
    # df=df.dropna()
    
    s=20431
    e=131024
    x=((x_interpolation*1000000)/x_factor)[s:e]
    y=((y_interpolation*1000000)/y_factor)[s:e]
    z=((z_interpolation*1000000)/z_factor)[s:e]
    
    
    time=np.linspace(0,len(x)-1,len(x))*(1/fs)
    
    
    
    
    # plt.plot(new_t[s:e],x,label='x')
    # plt.plot(new_t[s:e],y,label='y')
    # plt.plot(new_t[s:e],z,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('pixels')
    
    
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    # plt.plot(new_t[s:e],x_interpolation[s:e]*1000,label='x')
    # plt.plot(new_t[s:e],y_interpolation[s:e]*1000,label='y')
    # plt.plot(new_t[s:e],z_interpolation[s:e]*1000,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('mm')
    
    
    # plt.legend()
    # plt.grid()
    # plt.show()
    return np.round(x).astype(int),np.round(y).astype(int),np.round(z).astype(int)



# path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1243.mat'
# def read_data(path):
#     data = loadmat(path)
#     oct_volume = data['images']
#     return oct_volume

# original_volume=read_data(path)


def add_motion_to_volume(original_volume):
    x,y,z=get_movement()
    dims=original_volume.shape
    
    void_A_scan=np.zeros(512).astype(np.uint8)
    
    a_scan_counter=0
    
    try:
        volume_with_motion=np.zeros(dims)
        aa=[]
         
        for k in range(dims[2]): ### y
            for j in range(dims[1]): ### x  
                x_motion=j-x[a_scan_counter]
                y_motion=k-y[a_scan_counter]
                z_motion=z[a_scan_counter]
                a_scan_counter+=1
                
                if(x_motion<0 or x_motion>dims[1]-1 or y_motion<0 or y_motion>dims[2]-1 or np.abs(z_motion)>dims[0]-1):
                    A_scan_motion=void_A_scan
                else:
                    A_scan_motion=original_volume[:,x_motion,y_motion].copy()
                    z_abs=np.abs(z_motion)
                    
                    aa.append(z_motion)
                    if(z_motion<0):                
                        A_scan_motion=A_scan_motion[z_abs:]
                        A_scan_motion=np.concatenate((A_scan_motion,np.zeros(z_abs)))
                    else:
                        A_scan_motion=A_scan_motion[:len(A_scan_motion)-z_abs]
                        A_scan_motion=np.concatenate((np.zeros(z_abs),A_scan_motion))
                       
                       
                
                volume_with_motion[:,j,k]=A_scan_motion
        return volume_with_motion
    except:
        print('error') 
    
# volume_with_motion=add_motion_to_volume(original_volume)  
# import napari
# viewer = napari.view_image(volume_with_motion)

# viewer = napari.view_image(original_volume)


