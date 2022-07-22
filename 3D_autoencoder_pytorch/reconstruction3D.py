#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:34:27 2022

@author: diego
"""


import pickle
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from skimage.metrics import structural_similarity as ssim
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

bigger_sub_volumes_dim=(512,200,16)
original_volume_dim=(512,1000,100)
ngpu=1
model_path='autoencoder_for_reconstruction_BEST_MODEL_blue_noise_arch_1.pth'
mask_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/masks/mask_blue_noise_7575.pkl'
txt_test_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampling_bluenoise/test_volumes_paths.txt'
original_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_volumes/'


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


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


model_loaded= torch.load(model_path)



def reconstruct_volume(volume,reconstruction_model,sub_volumes_dim):
    
    h_div_factor = sub_volumes_dim[0]
    w_div_factor = sub_volumes_dim[1]
    d_div_factor = sub_volumes_dim[2]
    reconstructed_volume = np.zeros(original_volume_dim)
    overlap_pixels_w=50
    d_end=0
    for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        can_iterate_over_w=True
        last_iteration_w = False
        while can_iterate_over_w:
            h_end=0
            for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
                
    
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                sub_volume=np.expand_dims(sub_volume, axis=0)
                sub_volume=torch.from_numpy(sub_volume).to(device, dtype=torch.float)
                
                decoded_volume =  reconstruction_model(sub_volume).cpu().detach().numpy()
                squeezed_volume=np.squeeze(decoded_volume)
                if(w_end==0):
                    reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]=squeezed_volume[:,:,:]
                else:
                    reconstructed_volume[h_end:h_end+h_div_factor,w_end+30:w_end+w_div_factor,d_end:d_end+d_div_factor]=squeezed_volume[:,30:,:]

                h_end=h_end+h_div_factor
                if(h_end+h_div_factor>volume.shape[0]):
                    h_end=volume.shape[0]-h_div_factor
                    
            w_end=w_end+w_div_factor-overlap_pixels_w
            if(last_iteration_w):
                can_iterate_over_w=False
            if(w_end+w_div_factor>=volume.shape[1]):
                w_end=volume.shape[1]-w_div_factor
                last_iteration_w = True

                
        d_end=d_end+d_div_factor
        if(d_end+d_div_factor>volume.shape[2]):
            d_end=volume.shape[2]-d_div_factor
            
    return reconstructed_volume



def normalize(volume):
    max_value=(np.max(volume.astype(np.float32))/2)
    normalized_volume=(volume.astype(np.float32)-max_value)/max_value
    return normalized_volume,max_value

    
def compute_MAE(original,reconstruction):
    MAE=np.mean(np.abs(original - reconstruction))
    return MAE

def compute_RMSE(original,reconstruction):
    MSE=np.square(np.subtract(original,reconstruction)).mean()
    RMSE= np.sqrt(MSE)
    return RMSE
    
def compute_PSNR(original,reconstruction,bit_representation=8):
    MSE=np.square(np.subtract(original,reconstruction)).mean()
    MAXI=np.power(2,bit_representation)-1
    PSNR=20*np.log10(MAXI)-10*np.log10(MSE)
    return PSNR

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



reconstruction_model = Reconstruction_model(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    reconstruction_model = nn.DataParallel(reconstruction_model, list(range(ngpu)))


# Apply the trained weights_init 
reconstruction_model.apply(weights_init)

summary(reconstruction_model, bigger_sub_volumes_dim)




mask_blue_noise=load_obj(mask_path)


with open(txt_test_path) as f:
    lines = f.readlines()
test_volume_paths=[ original_volumes_path+name.split('/')[-1].split('.')[0]+'.pkl' for name in lines]    


PSNR_list=[]
RMSE_list=[]
MAE_list=[]
SSIM_list=[]
for i,test_volume_path in enumerate(test_volume_paths):
    try:
        print(test_volume_path.split('/')[-1]+'---------->'+str(i))
        original_volume=load_obj(test_volume_path)
    
    
        sub_sampled_volume=np.multiply(mask_blue_noise,original_volume).astype(np.uint8)
        
        ######## Normalize matrix###############################
        sub_sampled_volume_normalized,max_value=normalize(sub_sampled_volume)
        
        bigger_reconstruction=reconstruct_volume(sub_sampled_volume_normalized,reconstruction_model,bigger_sub_volumes_dim)
        
        ######## Denormalize matrix###############################
        
        bigger_reconstruction=(bigger_reconstruction*max_value)+max_value
        bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
        
        PSNR=compute_PSNR(original_volume,bigger_reconstruction)
        RMSE=compute_RMSE(original_volume,bigger_reconstruction)
        MAE=compute_MAE(original_volume,bigger_reconstruction)
        SSIM=ssim(original_volume,bigger_reconstruction)
        
        PSNR_list.append(PSNR)
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        SSIM_list.append(SSIM)
        if(i%10==0):
            print('Generating video...')
            gap=np.zeros((512,50,100)).astype(np.uint8)
            comparative_volume=np.concatenate((original_volume,gap,bigger_reconstruction,gap,sub_sampled_volume),axis=1)
            make_video(comparative_volume,'comparative_reconstruction_random_mask_'+str(i))
    except:
        print('Dimension ERROR...')
        

print('PSNR AVG: ',np.mean(PSNR_list))
print('RMSE AVG: ',np.mean(RMSE_list))
print('MAE AVG: ',np.mean(MAE_list))
print('SSIM AVG: ',np.mean(SSIM_list))
