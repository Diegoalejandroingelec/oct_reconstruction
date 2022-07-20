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

sub_volumes_dim=(512,64,16)
bigger_sub_volumes_dim=(512,900,16)

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

ngpu=2
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# h5_dataset_test=HDF5Dataset(subsampled_volumes_path_testing,original_volumes_path_testing,'original_test')
# # Create the dataloader
# dataloader_test = torch.utils.data.DataLoader(h5_dataset_test, batch_size=1, shuffle=True, num_workers=1)



class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()

        layers = [32,32,32,32,32,32]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[s+1])
            ) if s%2!=0 else nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[s+1]),
                nn.Dropout(p=0.5)
            ) for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            ) if s%2!=0 else nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
                nn.Dropout(p=0.5)
            )  for s in range(len(layers) - 1)
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

        layers = [32,32,32,32,32,32]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[s+1])
            )  for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            )  for s in range(len(layers) - 1)
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

        return x

# Then later:
model_loaded= torch.load('autoencoder_for_reconstruction_BEST_MODEL.pth')
#model_loaded= torch.load('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/GAN/generator_1.pth')

volume_dim=(512,1000,100)

def reconstruct_volume(volume,reconstruction_model,sub_volumes_dim):
    
    h_div_factor = sub_volumes_dim[0]
    w_div_factor = sub_volumes_dim[1]
    d_div_factor = sub_volumes_dim[2]
    reconstructed_volume = np.zeros(volume_dim)
    
    d_end=0
    for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        overlap_w=False
        for w in range(int(np.ceil(volume.shape[1]/w_div_factor))):
            h_end=0
            for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
                
    
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                sub_volume=np.expand_dims(sub_volume, axis=0)
                sub_volume=torch.from_numpy(sub_volume).to(device, dtype=torch.float)
                
                decoded_volume =  reconstruction_model(sub_volume).cpu().detach().numpy()
                print(decoded_volume.shape)
                if(not overlap_w):
                    reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]=decoded_volume
                else:
                    reconstructed_volume[h_end:h_end+h_div_factor,w_end+50:w_end+w_div_factor,d_end:d_end+d_div_factor]=decoded_volume[:,50:,:]
                
                h_end=h_end+h_div_factor
                if(h_end+h_div_factor>volume.shape[0]):
                    h_end=volume.shape[0]-h_div_factor
                    
            w_end=w_end+w_div_factor
            if(w_end+w_div_factor>volume.shape[1]):
                w_end=volume.shape[1]-w_div_factor
                overlap_w=True
                
        d_end=d_end+d_div_factor
        if(d_end+d_div_factor>volume.shape[2]):
            d_end=volume.shape[2]-d_div_factor
            
    return reconstructed_volume



#original_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/train/Farsiu_Ophthalmology_2013_AMD_Subject_1101.pkl')
#mask_blue_noise=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/masks/mask_blue_noise_7575.pkl')

original_volume=load_obj('Farsiu_Ophthalmology_2013_AMD_Subject_1101.pkl')
mask_blue_noise=load_obj('mask_blue_noise_7575.pkl')


sub_sampled_volume=np.multiply(mask_blue_noise,original_volume).astype(np.uint8)
def normalize(volume):
    return (volume.astype(np.float32)-(np.max(volume.astype(np.float32))/2))/(np.max(volume.astype(np.float32))/2)

######## Normalize matrix###############################
sub_sampled_volume_normalized=normalize(sub_sampled_volume)

reconstructed_volume_supreme=reconstruct_volume(sub_sampled_volume_normalized,model_loaded,sub_volumes_dim)

######## Denormalize matrix###############################


reconstructed_volume_supreme=(reconstructed_volume_supreme*127)+127

reconstructed_volume_supreme = reconstructed_volume_supreme.astype(np.uint8)

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
    
    
#make_video(sub_sampled_volume,'sub_sampled_blue_boise_23%')
make_video(reconstructed_volume_supreme,'reconstructed_volume_75%_512x64x16_without')

#make_video(original_volume,'ORIGINAL_TEST_75%')
#make_video(sub_sampled_volume,'sub_sampled_blue_boise_75%')
#make_video(reconstructed_volume_supreme,'reconstructed_volume_75%_512x64x16_blue_noise')



# signal=np.mean(original_volume)
# noise=np.std(original_volume)
# SNR_original=20*np.log10(signal/noise)


# signal_reconstructed=np.mean(reconstructed_volume_supreme)
# noise_reconstructed=np.std(reconstructed_volume_supreme)
# SNR_reconstructed=20*np.log10(signal_reconstructed/noise_reconstructed)
# print('SNR_RECONSTRUCTION:', SNR_reconstructed)



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


model_params = model_loaded.parameters()
param_list=[]
for weights in model_params:
    param_list.append(weights) 

global weights_index

weights_index=2
def weights_init(m):
    global weights_index
    classname = m.__class__.__name__  
    if(classname=='Conv3d' or classname=='ConvTranspose3d' or classname=='BatchNorm3d'):
        print(classname)
        print(weights_index-2)
        m.weight=param_list[weights_index-2]
        print(weights_index-1)
        m.bias=param_list[weights_index-1]
        weights_index+=2
        
        

class Reconstruction_model(nn.Module):
    def __init__(self,ngpu):
        super(Reconstruction_model,self).__init__()

        layers = [32,32,32,32,32,32]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[s+1])
            )  for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            )  for s in range(len(layers) - 1)
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

        return x


# Create the Discriminator
reconstruction_model = Reconstruction_model(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    reconstruction_model = nn.DataParallel(reconstruction_model, list(range(ngpu)))


# Apply the trained weights_init 
reconstruction_model.apply(weights_init)



# reconstruction_model_params = reconstruction_model.parameters()
# param_list1=[]
# for weights1 in reconstruction_model_params:
#     param_list1.append(weights1) 

summary(reconstruction_model, bigger_sub_volumes_dim)


######## Normalize matrix###############################
sub_sampled_volume_normalized=normalize(sub_sampled_volume)

bigger_reconstruction=reconstruct_volume(sub_sampled_volume_normalized,reconstruction_model,bigger_sub_volumes_dim)

######## Denormalize matrix###############################

bigger_reconstruction=(bigger_reconstruction*127.5)+127.5

bigger_reconstruction = bigger_reconstruction.astype(np.uint8)

make_video(bigger_reconstruction,'bigger_reconstruction_blue_noise_75')
