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
import matplotlib.pyplot as plt
import h5py

def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()

bigger_sub_volumes_dim=(512,64,16)
original_volume_dim=(512,1000,100)
ngpu=2
model_path='BEST_MODEL.pth'
mask_path='mask_random75.pkl'
txt_test_path='test_volumes_paths.txt'
original_volumes_path='../OCT_ORIGINAL_VOLUMES/'

# mask_path='mask_blue_noise_7575.pkl'
# txt_test_path='../../oct_data_blue_noise/test_volumes_paths.txt'
# original_volumes_path='../../OCT_ORIGINAL_VOLUMES/'

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Autoencoder(nn.Module):
    def __init__(self,ngpu):
        super(Autoencoder,self).__init__()

        layers = [16,16,16,16,16]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=9,padding='same'),
            nn.PReLU(),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.PReLU(),
                nn.BatchNorm3d(layers[s+1])
            )  for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.PReLU(),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            )  for s in range(len(layers) - 1)
        )

        self.output = nn.Sequential(
            nn.Conv3d(layers[0],1,kernel_size=9,padding='same'),
            nn.Tanh()
        )


    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = self.input(x)

        x_skip_connections=[]
        for i,j in enumerate(self.encoder):
            x = self.encoder[i](x)
            x_skip_connections.append(x)

        for i,j in enumerate(self.decoder):
            x = self.decoder[i](x)
            if(i!=len(x_skip_connections)-1):
                x = torch.add(x, x_skip_connections[len(x_skip_connections)-2-i])
        x = self.output(x)

        return x


model_loaded= torch.load(model_path)

def reconstruct_volume_batches(volume,reconstruction_model,sub_volumes_dim):
    batch_size_for_inference=1
    batch_for_inference=[]
    batch_metadata_for_reconstruction=[]
    h_div_factor = sub_volumes_dim[0]
    w_div_factor = sub_volumes_dim[1]
    d_div_factor = sub_volumes_dim[2]
    reconstructed_volume = np.zeros(original_volume_dim)
    overlap_pixels_w=30
    d_end=0
    for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        can_iterate_over_w=True
        last_iteration_w = False
        while can_iterate_over_w:
            h_end=0
            for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
        
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                batch_for_inference.append(sub_volume)
       
                if(w_end==0):
                    
                    data_for_reconstruction_by_chunks={
                        "coordinates_reconstructed":{
                            "h":(h_end,h_end+h_div_factor),
                            "w":(w_end,w_end+w_div_factor),
                            "d":(d_end,d_end+d_div_factor)
                            }
                        }
                    batch_metadata_for_reconstruction.append(data_for_reconstruction_by_chunks)
                    #reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]=squeezed_volume[:,:,:]
                else:
                    data_for_reconstruction_by_chunks={
                        "coordinates_reconstructed":{
                            "h":(h_end,h_end+h_div_factor),
                            "w":(w_end+15,w_end+w_div_factor),
                            "d":(d_end,d_end+d_div_factor)
                            }
                        }
                    batch_metadata_for_reconstruction.append(data_for_reconstruction_by_chunks)
                    

                if len(batch_for_inference)==batch_size_for_inference:
                    batch_for_inference=np.array(batch_for_inference)
                    batch_for_inference=torch.from_numpy(batch_for_inference).to(device, dtype=torch.float)
                    reconstructed_batch =  reconstruction_model(batch_for_inference).cpu().detach().numpy()
                    
                    
                    squeezed_volumes=np.squeeze(reconstructed_batch,1)
                    for index,vol in enumerate(squeezed_volumes):
                        h_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['h'][0]
                        h_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['h'][1]
                        
                        w_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['w'][0]
                        w_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['w'][1]
                        
                        d_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['d'][0]
                        d_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['d'][1]
                        
                        if(w_start==0):
                            reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,:,:]
                        else:
                            reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,15:,:]
                    
                    batch_metadata_for_reconstruction=[]
                    batch_for_inference=[]

                

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
            
    if len(batch_for_inference)>0:
        batch_for_inference=np.array(batch_for_inference)
        batch_for_inference=torch.from_numpy(batch_for_inference).to(device, dtype=torch.float)
        reconstructed_batch =  reconstruction_model(batch_for_inference).cpu().detach().numpy()
        
        
        squeezed_volumes=np.squeeze(reconstructed_batch)
        for index,vol in enumerate(squeezed_volumes):
            h_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['h'][0]
            h_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['h'][1]
            
            w_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['w'][0]
            w_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['w'][1]
            
            d_start=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['d'][0]
            d_finish=batch_metadata_for_reconstruction[index]['coordinates_reconstructed']['d'][1]
            
            if(w_start==0):
                reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,:,:]
            else:
                reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,15:,:]
   
    return reconstructed_volume


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
                    print('h: ', (h_end,h_end+h_div_factor))
                    print('w: ', (w_end,w_end+w_div_factor))
                    print('d: ', (d_end,d_end+d_div_factor))
                    print('')
                    reconstructed_volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]=squeezed_volume[:,:,:]
                else:
                    print('h: ', (h_end,h_end+h_div_factor))
                    print('w: ', (w_end+30,w_end+w_div_factor))
                    print('d: ', (d_end,d_end+d_div_factor))
                    print('')
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
    mae = nn.L1Loss()
    x=torch.from_numpy(original.astype('float'))
    y=torch.from_numpy(reconstruction.astype('float'))
    MAE= mae(x,y).item()
    return MAE




def compute_RMSE(original,reconstruction):
    mse = nn.MSELoss()
    x=torch.from_numpy(original.astype('float'))
    y=torch.from_numpy(reconstruction.astype('float'))    
    MSE=mse(x, y)
    MSE=MSE.item()
    RMSE= np.sqrt(MSE)
    return RMSE
    
def compute_PSNR(original,reconstruction,bit_representation=8):
    mse = nn.MSELoss()
    x=torch.from_numpy(original.astype('float'))
    y=torch.from_numpy(reconstruction.astype('float'))    
    MSE=mse(x, y)
    MSE=MSE.item()
    MAXI=np.power(2,bit_representation)-1
    PSNR=20*np.log10(MAXI)-10*np.log10(MSE)
    return PSNR

model_params = model_loaded.parameters()
param_list=[]
for weights in model_params:
    param_list.append(weights) 

global weights_index

weights_index=0
def weights_init(m):
    global weights_index
    classname = m.__class__.__name__
    if(classname=='Conv3d' or classname=='ConvTranspose3d' or classname=='BatchNorm3d'):
        print(classname)
        weights_index+=2
        print(weights_index-2)
        m.weight=param_list[weights_index-2]
        print(weights_index-1)
        m.bias=param_list[weights_index-1]
    elif(classname=='PReLU'):
        print(classname)
        weights_index+=1
        print(weights_index-1)
        print(param_list[weights_index-1].shape)
        m.weight=param_list[weights_index-1]

class Reconstruction_model(nn.Module):
    def __init__(self,ngpu):
        super(Reconstruction_model,self).__init__()

        layers = [16,16,16,16,16]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=9,padding='same'),
            nn.PReLU(),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.PReLU(),
                nn.BatchNorm3d(layers[s+1])
            )  for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.PReLU(),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            )  for s in range(len(layers) - 1)
        )

        self.output = nn.Sequential(
            nn.Conv3d(layers[0],1,kernel_size=9,padding='same'),
            nn.Tanh()
        )


    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = self.input(x)

        x_skip_connections=[]
        for i,j in enumerate(self.encoder):
            x = self.encoder[i](x)
            x_skip_connections.append(x)

        for i,j in enumerate(self.decoder):
            x = self.decoder[i](x)
            if(i!=len(x_skip_connections)-1):
                x = torch.add(x, x_skip_connections[len(x_skip_connections)-2-i])
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


PSNR_list_sub=[]
RMSE_list_sub=[]
MAE_list_sub=[]
SSIM_list_sub=[]


for i,test_volume_path in enumerate(test_volume_paths):
    try:
        print(test_volume_path.split('/')[-1]+'---------->'+str(i))
        original_volume=load_obj(test_volume_path)

        sub_sampled_volume=np.multiply(mask_blue_noise,original_volume).astype(np.uint8)
        ######## Normalize matrix###############################
        sub_sampled_volume_normalized,max_value=normalize(sub_sampled_volume)
        
        bigger_reconstruction=reconstruct_volume_batches(sub_sampled_volume_normalized,reconstruction_model,bigger_sub_volumes_dim)
        
        ######## Denormalize matrix###############################
        
        bigger_reconstruction=(bigger_reconstruction*max_value)+max_value
        bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
        

        mask_blue_noise_prima= (~mask_blue_noise.astype(bool)).astype(int)

        bigger_reconstruction=np.multiply(mask_blue_noise_prima,bigger_reconstruction).astype(np.uint8)
        bigger_reconstruction=bigger_reconstruction+sub_sampled_volume

        PSNR=compute_PSNR(original_volume,bigger_reconstruction)
        RMSE=compute_RMSE(original_volume,bigger_reconstruction)
        MAE=compute_MAE(original_volume,bigger_reconstruction)
        SSIM=ssim(original_volume.astype(np.uint8),bigger_reconstruction.astype(np.uint8))
        
        # PSNR_sub=compute_PSNR(original_volume,sub_sampled_volume)
        # RMSE_sub=compute_RMSE(original_volume,sub_sampled_volume)
        # MAE_sub=compute_MAE(original_volume,sub_sampled_volume)
        # SSIM_sub=ssim(original_volume.astype(np.uint8),sub_sampled_volume.astype(np.uint8))
        
        PSNR_list.append(PSNR)
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        SSIM_list.append(SSIM)
        
        # PSNR_list_sub.append(PSNR_sub)
        # RMSE_list_sub.append(RMSE_sub)
        # MAE_list_sub.append(MAE_sub)
        # SSIM_list_sub.append(SSIM_sub)
        plt.imshow(sub_sampled_volume[:,:,0], cmap="gray")
        plt.show()
        plt.savefig('subsampled.png')
        if(i%10==0):
            print('Generating video...')
            gap=np.zeros((512,50,100)).astype(np.uint8)
            comparative_volume=np.concatenate((original_volume,gap,bigger_reconstruction,gap,sub_sampled_volume),axis=1)
            make_video(comparative_volume,'comparative_reconstruction_'+str(i))
    except Exception as e:
        print(e)
        print('Dimension ERROR...')

print('PSNR AVG: ',np.mean(PSNR_list))
print('RMSE AVG: ',np.mean(RMSE_list))
print('MAE AVG: ',np.mean(MAE_list))
print('SSIM AVG: ',np.mean(SSIM_list))


# print('PSNR AVG: ',np.mean(PSNR_list_sub))
# print('RMSE AVG: ',np.mean(RMSE_list_sub))
# print('MAE AVG: ',np.mean(MAE_list_sub))
# print('SSIM AVG: ',np.mean(SSIM_list_sub))
