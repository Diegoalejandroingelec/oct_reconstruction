#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:53:31 2022

@author: diego
"""
import numpy as np
import pickle
import cv2
import torch
import torch.nn as nn


ngpu=1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

sub_volumes_dim=(512,64,16)


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()


def create_blue_noise_mask(expected_dims,kernel,subsampling_percentage):
    
    blue_noise_cube_normalized=kernel/np.max(kernel)
    blue_noise_cube_normalized_shape=blue_noise_cube_normalized.shape
    
    axis_0=int(np.ceil(expected_dims[0]/blue_noise_cube_normalized_shape[0]))
    axis_1=int(np.ceil(expected_dims[1]/blue_noise_cube_normalized_shape[1]))
    axis_2=int(np.ceil(expected_dims[2]/blue_noise_cube_normalized_shape[2]))
    
    
    
    
    concat1 = np.concatenate(tuple([blue_noise_cube_normalized for i in range(axis_0)]), axis=0)
    concat2 =np.concatenate(tuple([concat1 for i in range(axis_1)]),axis=1)
    concat3 =np. concatenate(tuple([concat2 for i in range(axis_2)]),axis=2)
    
    blue_noise_mask = concat3[0:expected_dims[0],0:expected_dims[1],0:expected_dims[2]]
    
    
    
    binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
    binary_blue_noise_mask = binary_blue_noise_mask*1
    total = binary_blue_noise_mask.sum()
    
    
    missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
    print(missing_data)
    
    return blue_noise_mask,binary_blue_noise_mask


blue_noise_cube = np.load('bluenoisecube.npy')

blue_noise_mask,binary_blue_noise_mask= create_blue_noise_mask(expected_dims=(512,1000,100),
                                                               kernel=blue_noise_cube,
                                                               subsampling_percentage=0.75)




    
    
original_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')
sub_sampled_data = np.multiply(binary_blue_noise_mask,original_volume)


class Autoencoder(nn.Module):
    def __init__(self,ngpu):
        super(Autoencoder,self).__init__()

        layers = [32,32,16,16]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.LayerNorm([layers[0],sub_volumes_dim[0],sub_volumes_dim[1],sub_volumes_dim[2]])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.ReLU(),
                nn.LayerNorm([layers[s+1],sub_volumes_dim[0]-2*(s+1),sub_volumes_dim[1]-2*(s+1),sub_volumes_dim[2]-2*(s+1)])
            ) for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.ReLU(),
                nn.LayerNorm([layers[len(layers)-2-s],sub_volumes_dim[0]-2*(len(layers)-2-s),sub_volumes_dim[1]-2*(len(layers)-2-s),sub_volumes_dim[2]-2*(len(layers)-2-s)])
            ) for s in range(len(layers) - 1)
        )

        self.output = nn.Sequential(
            nn.Conv3d(layers[0],1,kernel_size=3,padding='same')
        )


    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = self.input(x)

        for i,j in enumerate(self.encoder):
            x = self.encoder[i](x)

        for i,j in enumerate(self.decoder):
            x = self.decoder[i](x)
        
        x = self.output(x)

        x = torch.squeeze(x)
        return x

# Then later:
model_loaded= torch.load('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/3D_autoencoder_pytorch/autoencoder_for_reconstruction.pth')
volume_dim=(512,1000,100)

def reconstruct_volume(volume,reconstruction_model,w_div_factor,h_div_factor,d_div_factor):
    reconstructed_volume = np.zeros(volume_dim)
    
    d_end=0
    for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        for w in range(int(np.ceil(volume.shape[1]/w_div_factor))):
            h_end=0
            for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
                
    
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                sub_volume=np.expand_dims(sub_volume, axis=0)
                sub_volume=torch.from_numpy(sub_volume).to(device, dtype=torch.float)
                
                decoded_volume =  reconstruction_model(sub_volume).cpu().detach().numpy()
                
                reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]=decoded_volume
                
                
                h_end=h_end+h_div_factor
                if(h_end+h_div_factor>volume.shape[0]):
                    h_end=volume.shape[0]-h_div_factor
                    
            w_end=w_end+w_div_factor
            if(w_end+w_div_factor>volume.shape[1]):
                w_end=volume.shape[1]-w_div_factor
                
        d_end=d_end+d_div_factor
        if(d_end+d_div_factor>volume.shape[2]):
            d_end=volume.shape[2]-d_div_factor
            
    return reconstructed_volume



reconstructed_volume_supreme=reconstruct_volume(sub_sampled_data,model_loaded,64,512,16)
reconstructed_volume_supreme=reconstructed_volume_supreme.astype(np.uint8)









































#make_video(original_volume,'Original_volume')
#make_video(sub_sampled_data,'blue_noise_subsampled_99%')
