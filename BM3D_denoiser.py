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
import cv2

def BM3D_denoiser(volume,sigma_psd=0.1):
    start = time.process_time()
    noisy_img=  img_as_float(volume)
    BM3D_denoised=bm3d.bm3d(noisy_img,
                            sigma_psd,
                            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
    print('TIME ELAPSED FOR DENOISING VOLUME:', time.process_time() - start, 's')
    return (BM3D_denoised*255).astype(np.uint8)


    
def median_filter_3D(volume,threshold,window_size):
    start = time.process_time()
    filtered_volume=np.zeros(volume.shape)
    volume=np.transpose(volume,(2,0,1))
    print("DENOISING VOLUME...")
    for index,image in enumerate(volume):
        padding_size=window_size//2
        img_with_padding=cv2.copyMakeBorder(image,
                                            padding_size,
                                            padding_size,
                                            padding_size,
                                            padding_size,
                                            cv2.BORDER_REPLICATE)
        filtered_image=np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                central_value=image[i,j]
                if(central_value<threshold):
                    values=img_with_padding[i:i+window_size,j:j+window_size]
                    median=np.median(values)
                    filtered_image[i,j]=median
                else:
                    filtered_image[i,j]=central_value
        filtered_volume[:,:,index]=filtered_image
    print('TIME ELAPSED FOR DENOISING VOLUME:', time.process_time() - start, 's')
    return filtered_volume.astype(np.uint8)



# import pickle
# def load_obj(name):
#     with open( name, 'rb') as f:
#         return pickle.load(f)

# original_volume =load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_volumes/Farsiu_Ophthalmology_2013_AMD_Subject_1001.pkl')

# median2=median_filter_3D(original_volume,40,5)

# cv2.imshow('original Image',original_volume[:,:,0])
# cv2.imshow('denoised Image2',median2[:,:,0].astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import matplotlib.pyplot as plt

# plt.plot(original_volume[:,0,0])
# plt.plot(median2[:,0])
# plt.show()