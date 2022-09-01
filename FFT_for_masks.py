#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:13:02 2022

@author: diego
"""
#from Generate_shifting_blue_noise import generate_shifting_blue_noise,generate_binary_blue_noise_mask,generate_shifting_blue_noise_all_directions
import numpy as np 
import napari
#import matplotlib.pyplot as plt

# def create_blue_noise_mask_1(expected_dims,subsampling_percentage):
#     blue_noise=generate_shifting_blue_noise(expected_dims)
#     mask=generate_binary_blue_noise_mask(blue_noise,subsampling_percentage)
#     return mask

# def create_blue_noise_mask_2(expected_dims,subsampling_percentage):
#     blue_noise=generate_shifting_blue_noise_all_directions(expected_dims)
#     mask=generate_binary_blue_noise_mask(blue_noise,subsampling_percentage)
#     return mask


# mask=create_blue_noise_mask_2(expected_dims=(512,1000,100),subsampling_percentage=0.75)

# mask=np.load('./blue_noise_cubes/blue_noise_vube_64_64_64.npy')
# mask=np.concatenate((mask,mask),axis=0)
# mask=np.concatenate((mask,mask),axis=1)
# mask=np.concatenate((mask,mask),axis=2)
#mask=np.load('./example_masks/BLUE_NOISE_SHIFTING_TWO_DIRECTIONS.npy')
# mask=np.load('./example_masks/RISLEY_4_PRISMS.npy')
#mask=np.load('./example_masks/RISLEY_4_PRISMS_GAUSSIAN_100.npy')
# mask=np.load('./example_masks/RISLEY_4_PRISMS_GAUSSIAN_200.npy')
# mask=np.load('./example_masks/RISLEY_4_PRISMS_SEMI_GAUSSIAN_100.npy')
mask=np.load('./example_masks/BLUE_NOISE_SHIFTING_ONE_DIRECTION.npy')




img=mask*255
# f = np.fft.fft2(img.astype(np.float32))
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

DFT=np.fft.fftshift(np.fft.fftn(img))/float(np.size(img));
# for n in range(64):
#     plt.imshow(np.abs(DFT[:,:,n]),
#                   cmap="viridis",
#                   interpolation="nearest",
#                   vmin=0.0,
#                   vmax=np.percentile(np.abs(DFT),99));
#     plt.colorbar();
#     plt.show()

magnitude=20*np.log(np.abs(DFT))
viewer = napari.view_image(magnitude)
viewer = napari.view_image(mask*255)

