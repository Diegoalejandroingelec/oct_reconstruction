#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:16:50 2022

@author: diego
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import cv2

reconstruction_model = keras.models.load_model('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/models/model_75')
volume_dim=(512,1000,100)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)


def reconstruct_volume(volume,reconstruction_model):
    w_div_factor = 64
    h_div_factor = 64
    w_end=0
    reconstructed_volume = np.zeros(volume_dim)
    for w in range(int(np.ceil(volume.shape[1]/w_div_factor))):
        h_end=0
        for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
            

            sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,:]
            sub_volume=np.expand_dims(sub_volume, axis=0)
            sub_volume=np.expand_dims(sub_volume, axis=4)
            
            encoded_volume = reconstruction_model.encoder(sub_volume).numpy()
            decoded_volume = reconstruction_model.decoder(encoded_volume).numpy()
            
            decoded_volume=np.squeeze(decoded_volume)
            reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,:]=decoded_volume
            
            
            h_end=h_end+h_div_factor
            
            
            if(h_end+h_div_factor>volume.shape[0]):
                h_end=volume.shape[0]-h_div_factor
                
            
            
        w_end=w_end+w_div_factor
        if(w_end+w_div_factor>volume.shape[1]):
            w_end=volume.shape[1]-w_div_factor
            
    return reconstructed_volume




sub_sampled_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/subsampled_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1001_subsampled_75.pkl')
original_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1001.pkl')

reconstructed_volume=reconstruct_volume(sub_sampled_volume,reconstruction_model)


reconstructed_volume=reconstructed_volume*255
reconstructed_volume=reconstructed_volume.astype('uint8')


cv2.imshow('Original Volume',original_volume[:,:,0])
cv2.imshow('Subsampled Volume',sub_sampled_volume[:,:,0])
for i in range(100):
    cv2.imshow('Reconstructed Volume',reconstructed_volume[:,:,i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()




