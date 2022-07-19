#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:34:27 2022

@author: diego
"""

import h5py
import pickle
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torchvision.utils as vutils
sub_volumes_dim=(512,64,16)


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
    
class HDF5Dataset(data.Dataset):

    def __init__(self, file_path, ground_truth_path, prefix_for_test, transform=None):
        super().__init__()
        self.file_path = file_path
        self.ground_truth_path = ground_truth_path
        self.transform = transform
        self.prefix_for_test=prefix_for_test

    def __getitem__(self, index):
        # get data
        x,name = self.get_data(index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_ground_truth(name)
        
        if self.transform:
            y = self.transform(y)
        else:
            y = torch.from_numpy(y)
        return (x, y)
    
    def get_ground_truth(self,reference_name):
        f_gt = h5py.File(self.ground_truth_path, 'r')
        
        name = self.prefix_for_test+'_'+'_'.join(reference_name.split('_')[-3:])
        value=np.array(f_gt.get(name))
        f_gt.close()
        return value
    
    def __len__(self):
        return self.get_info()
    
    def get_data(self,index):
        f = h5py.File(self.file_path, 'r')
        name=list(f.keys())[index]
        value=np.array(f.get(name))
        f.close()
        return value,name
    
    def get_info(self):
        f = h5py.File(self.file_path, 'r')
        info=len(list(f.keys()))
        f.close()
        return info

subsampled_volumes_path_testing='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/testing_subsampled_volumes.h5'
original_volumes_path_testing='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/testing_ground_truth.h5'

ngpu=1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# h5_dataset_test=HDF5Dataset(subsampled_volumes_path_testing,original_volumes_path_testing,'original_test')
# # Create the dataloader
# dataloader_test = torch.utils.data.DataLoader(h5_dataset_test, batch_size=1, shuffle=True, num_workers=1)



class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()

        layers = [32,32,32,32]
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
            nn.Conv3d(layers[0],1,kernel_size=3,padding='same'),
            nn.Tanh()
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
#model_loaded= torch.load('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/3D_autoencoder_pytorch/autoencoder_for_reconstruction.pth')
model_loaded= torch.load('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/GAN/generator.pth')

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


sub_sampled_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/subsampled_23/subsapled_Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')
original_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')

def normalize(volume):
    return (volume-np.max(volume))#/np.max(volume)

sub_sampled_volume_normalized=normalize(sub_sampled_volume)
reconstructed_volume_supreme=reconstruct_volume(sub_sampled_volume,model_loaded,64,512,16)
reconstructed_volume_supreme=(reconstructed_volume_supreme*255)
reconstructed_volume_supreme=reconstructed_volume_supreme.astype(np.uint8)

sub_sampled_volume=sub_sampled_volume*255
sub_sampled_volume=sub_sampled_volume.astype(np.uint8)

#reconstructed_volume_supreme=reconstructed_volume_supreme.astype(np.uint8)
import cv2
def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()

# make_video(original_volume,'original_volume_75%')
# make_video(sub_sampled_volume,'subsampled_volume_75%')
# make_video(reconstructed_volume_supreme,'reconstructed_volume_75%_512x64x16_GAN')






# test_losses=[]
# criterion = nn.MSELoss()
# for i, data_test in enumerate(dataloader_test, 0):  
#     inputs = data_test[0].to(device, dtype=torch.float)
#     targets = data_test[1].to(device, dtype=torch.float)
#     # compute the model output
#     reconstructions_test = model_loaded(inputs)
#     # calculate loss
#     loss_test = criterion(reconstructions_test, torch.squeeze(targets))
#     test_losses.append(loss_test.item())
#     if i % 50 == 0:
#         print(i)

# np.mean(test_losses)

# summary(model_loaded, sub_volumes_dim)


# model_params = model_loaded.parameters()
# param_list=[]
# for weights in model_params:
#     param_list.append(weights) 

# global weights_index

# weights_index=2
# def weights_init(m):
#     global weights_index
#     classname = m.__class__.__name__  
#     if(classname=='Conv3d' or classname=='LayerNorm' or classname=='ConvTranspose3d'):
#         print(classname)
#         print(weights_index-2)
#         m.weight=param_list[weights_index-2]
#         print(weights_index-1)
#         m.bias=param_list[weights_index-1]
#         weights_index+=2
        
        

# class Autoencoder_for_reconstruction(nn.Module):
#     def __init__(self,ngpu):
#         super(Autoencoder_for_reconstruction,self).__init__()

#         layers = [32,32,16,16]
#         self.ngpu = ngpu
        
#         self.input = nn.Sequential(
#             nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
#             nn.ReLU(),
#             nn.LayerNorm([layers[0],sub_volumes_dim[0],sub_volumes_dim[1],sub_volumes_dim[2]])
#         )

#         self.encoder = nn.ModuleList(
#             nn.Sequential(
#                 nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
#                 nn.ReLU(),
#                 nn.LayerNorm([layers[s+1],sub_volumes_dim[0]-2*(s+1),sub_volumes_dim[1]-2*(s+1),sub_volumes_dim[2]-2*(s+1)])
#             ) for s in range(len(layers) - 1)
#         )

#         self.decoder = nn.ModuleList(
#             nn.Sequential(
#                 nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
#                 nn.ReLU(),
#                 nn.LayerNorm([layers[len(layers)-2-s],sub_volumes_dim[0]-2*(len(layers)-2-s),sub_volumes_dim[1]-2*(len(layers)-2-s),sub_volumes_dim[2]-2*(len(layers)-2-s)])
#             ) for s in range(len(layers) - 1)
#         )

#         self.output = nn.Sequential(
#             nn.Conv3d(layers[0],1,kernel_size=3,padding='same')
#         )


#     def forward(self,x):
#         x = torch.unsqueeze(x,1)
#         x = self.input(x)

#         for i,j in enumerate(self.encoder):
#             x = self.encoder[i](x)

#         for i,j in enumerate(self.decoder):
#             x = self.decoder[i](x)
        
#         x = self.output(x)

#         x = torch.squeeze(x)
#         return x


# # Create the Discriminator
# reconstruction_model = Autoencoder_for_reconstruction(ngpu).to(device)



# # Apply the weights_init function to randomly initialize all weights
# #  to mean=0, stdev=0.2.
# reconstruction_model.apply(weights_init)



# reconstruction_model = reconstruction_model.parameters()
# param_list1=[]
# for weights1 in reconstruction_model:
#     param_list1.append(weights1) 