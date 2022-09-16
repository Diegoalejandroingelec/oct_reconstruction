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
    
    x = (b1.gen_random_walk(512)*10)
    y = (b2.gen_random_walk(512)*10)
    
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

# import napari

# viewer = napari.view_image(volume_with_motion)
#     plt.plot(x[0:i],y[0:i],c='b')
    
    
#     plt.grid()
#     plt.show()