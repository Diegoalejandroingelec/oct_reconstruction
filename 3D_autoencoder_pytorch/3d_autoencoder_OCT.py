#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:53:40 2022

@author: diego
"""

import h5py

import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torchvision.utils as vutils

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 4

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

sub_volumes_dim=(512,64,16)



class HDF5Dataset(data.Dataset):

    def __init__(self, file_path, ground_truth_path, transform=None):
        super().__init__()
        self.file_path = file_path
        self.ground_truth_path = ground_truth_path
        self.transform = transform

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
        name = 'original_train_'+ '_'.join(reference_name.split('_')[-3:])
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
    
def normalize(volume):
    return volume/np.max(volume)

subsampled_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/training_subsampled_volumes.h5'
original_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/training_ground_truth.h5'

h5_dataset=HDF5Dataset(subsampled_volumes_path,original_volumes_path)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(h5_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)





# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
class Autoencoder(nn.Module):
    def __init__(self,ngpu):
        super(Autoencoder,self).__init__()

        layers = [32,16,16,8]
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

# Create the generator
netG = Autoencoder(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

summary(netG, sub_volumes_dim)


criterion = nn.MSELoss()
optimizer = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


print("Starting Training Loop...")
# For each epoch
losses = []
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):  
        inputs = data[0].to(device, dtype=torch.float)
        targets = data[1].to(device, dtype=torch.float)
        
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        reconstructions = netG(inputs)
        # calculate loss
        loss = criterion(reconstructions, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader),loss.item()))
            
        losses.append(loss.item())
