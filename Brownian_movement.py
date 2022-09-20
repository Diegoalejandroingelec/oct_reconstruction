#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:28:14 2022

@author: diego
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat


class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
def add_motion_to_en_face_images(original_volume,plot_random_walk):
    b1 = Brownian()
    b2 = Brownian()
    total_steps=0
    random_steps=50
    x=np.empty(shape=[0])
    y=np.empty(shape=[0])
    while(total_steps<512):
        random_x_walk=b1.gen_random_walk(random_steps)
        random_y_walk=b2.gen_random_walk(random_steps)
        
        x_factor=1
        y_factor=1
        
        
        x=np.concatenate((x,(random_x_walk)*x_factor))
        y=np.concatenate((y,(random_y_walk)*y_factor))
        total_steps+=random_steps
    
    x=x[0:512]
    y=y[0:512]
    if(plot_random_walk):
        plt.plot(x,y,c='b')
        plt.show()
        
    volume_with_motion=[]
    for i in range(len(x)):
        #print(i)
        img=original_volume[i,:,:]
        num_rows, num_cols = img.shape[:2]
        translation_matrix = np.float32([ [1,0,x[i]], [0,1,y[i] ]])   
        img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows)) 
        # cv2.imshow('Translation', img_translation)    
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
        volume_with_motion.append(img_translation)
        
    volume_with_motion=np.array(volume_with_motion)
    
    return volume_with_motion


# path='../oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1253.mat'
# def read_data(path):
#     data = loadmat(path)
#     oct_volume = data['images']
#     return oct_volume

# original_volume=read_data(path)

# volume_with_motion=add_motion_to_en_face_images(original_volume,plot_random_walk=True)
# viewer = napari.view_image(volume_with_motion)
# viewer = napari.view_image(original_volume)
#     plt.plot(x[0:i],y[0:i],c='b')
    
    
#     plt.grid()
#     plt.show()