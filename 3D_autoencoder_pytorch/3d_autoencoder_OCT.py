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
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

sub_volumes_dim=(512,64,16)


def save_obj(obj,path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class HDF5Dataset(data.Dataset):

    def __init__(self, file_path, ground_truth_path,prefix_for_test, transform=None):
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
        name = self.prefix_for_test+'_'+ '_'.join(reference_name.split('_')[-3:])
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
    return (volume.astype(np.float32)-(np.max(volume.astype(np.float32))/2))/(np.max(volume.astype(np.float32))/2)


#subsampled_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampling_bluenoise/training_blue_noise_subsampled_volumes.h5'
#original_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampling_bluenoise/training_blue_noise_ground_truth.h5'

subsampled_volumes_path='../../oct_data_blue_noise/training_blue_noise_subsampled_volumes.h5'
original_volumes_path='../../oct_data_blue_noise/training_blue_noise_ground_truth.h5'



h5_dataset=HDF5Dataset(subsampled_volumes_path,original_volumes_path,'original_train',normalize)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(h5_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# train_batch = next(iter(dataloader))

# plt.imshow(np.squeeze(np.array(train_batch[1][0,:,:,0].cpu())), cmap="gray")
# plt.imshow(np.squeeze(np.array(train_batch[0][0,:,:,0].cpu())), cmap="gray")

#subsampled_volumes_path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampling_bluenoise/testing_blue_noise_subsampled_volumes.h5'
#original_volumes_path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampling_bluenoise/testing_blue_noise_ground_truth.h5'


subsampled_volumes_path_test='../../oct_data_blue_noise/testing_blue_noise_subsampled_volumes.h5'
original_volumes_path_test='../../oct_data_blue_noise/testing_blue_noise_ground_truth.h5'

h5_dataset_test=HDF5Dataset(subsampled_volumes_path_test,original_volumes_path_test,'original_test',normalize)
# Create the dataloader
dataloader_test = torch.utils.data.DataLoader(h5_dataset_test, batch_size=1, shuffle=True, num_workers=workers)

# test_batch = next(iter(dataloader_test))

# plt.imshow(np.squeeze(np.array(test_batch[1][0,:,:,0].cpu())), cmap="gray")
# plt.imshow(np.squeeze(np.array(test_batch[0][0,:,:,0].cpu())), cmap="gray")


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
class Autoencoder(nn.Module):
    def __init__(self,ngpu):
        super(Autoencoder,self).__init__()

        layers = [32,32,16,16]
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

criterion_for_testing=nn.MSELoss()

optimizer = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


print("Starting Training Loop...")
# For each epoch
losses = []
losses_val=[]
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data_train in enumerate(dataloader, 0):  
        inputs = data_train[0].to(device, dtype=torch.float)
        targets = data_train[1].to(device, dtype=torch.float)
        
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        reconstructions = netG(inputs)
        # calculate loss
        loss = criterion(reconstructions, torch.unsqueeze(targets,1))
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader),loss.item()))
            
        losses.append(loss.item())

        
    test_losses=[]
    print('Evaluation...')
    for j, data_test in enumerate(dataloader_test, 0):  
         inputs_test = data_test[0].to(device, dtype=torch.float)
         targets_test = data_test[1].to(device, dtype=torch.float)
         # compute the model output
         reconstructions_test = netG(inputs_test)
         # calculate loss
         loss_test = criterion_for_testing(reconstructions_test, torch.unsqueeze(targets_test,1))
         test_losses.append(loss_test.item())
         if j % 5000 == 0:
             print(j)

    current_loss=np.mean(test_losses)
    print('VALIDATION LOSS: ',current_loss)
    if(len(losses_val)>0):
        min_val_loss=np.min(losses_val)
        if(current_loss<min_val_loss):
            torch.save(netG, 'autoencoder_for_reconstruction_BEST_MODEL_blue_noise_arch_1.pth')
    else:
        torch.save(netG, 'autoencoder_for_reconstruction_first_epoch_blue_noise_arch_1.pth')
        
    losses_val.append(current_loss)
    
    
save_obj(losses,'train_losses_blue_noise_arch1' )
save_obj(losses_val,'test_losses_blue_noise_arch1' )      
      
      
torch.save(netG, 'autoencoder_for_reconstruction_last_epoch_blue_noise_arch1.pth')

# def load_obj(name):
#     with open( name, 'rb') as f:
#         return pickle.load(f)
    
# L=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/3D_autoencoder_pytorch/test_losses_random_sub.pkl')    
# plt.figure(figsize=(10,5))
# plt.title("Autoencoder Loss During Training")
# plt.plot(L,label="autoencoder")

# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

###############################################################################
###################                                         ###################
###################                                         ###################
###################                                         ###################
###################          TESTING                        ###################
###################                                         ###################
###################                                         ###################
###################                                         ###################
###############################################################################                   

# subsampled_volumes_path_testing='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/testing_subsampled_volumes.h5'
# original_volumes_path_testing='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/data_train_autoencoder3D/testing_ground_truth.h5'


# h5_dataset_test=HDF5Dataset(subsampled_volumes_path_testing,original_volumes_path_testing)
# # Create the dataloader
# dataloader_test = torch.utils.data.DataLoader(h5_dataset_test, batch_size=2, shuffle=True, num_workers=1)


# real_batch = next(iter(dataloader_test))

# test_losses=[]
# for flo in dataloader_test:  
#     print(flo)
#     break
#     inputs = data[0].to(device, dtype=torch.float)
#     targets = data[1].to(device, dtype=torch.float)
#     # compute the model output
#     reconstructions_test = netG(inputs)
#     # calculate loss
#     loss_test = criterion(reconstructions_test, targets)
#     test_losses.append(loss_test.toitem())
#     if i % 50 == 0:
#         print(i)

# np.mean(test_losses)


# import cv2

# t=np.array(targets[0,:,:,:].cpu()).astype(np.uint8)
# inp=np.array(inputs[0,:,:,:].cpu()).astype(np.uint8)
# resul=model_loaded(inputs)
# resul=resul[0,:,:,:].cpu().detach().numpy().astype(np.uint8)


# t = t[:,:,3]
# inp=inp[:,:,3]
# resul=resul[:,:,3]


# cv2.imshow('Original Image',t)
# cv2.imshow('subsampled',inp)
# cv2.imshow('Reconstruction',resul)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
