#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:07:48 2022

@author: diego
"""
from GAN_models import Generator
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

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
model_path='./results/GAN_OCT/g_best.pth.tar'
mask_path='../../BLUE_NOISE_DATASET/mask_random_blue_noise175.pkl'
txt_test_path='../../BLUE_NOISE_DATASET/test_volumes_paths_blue_noise1.txt'
original_volumes_path='../../../OCT_ORIGINAL_VOLUMES/'

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
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
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


model=torch.load('./results/GAN_OCT/g_best.pth.tar')

generator = Generator(ngpu)
generator = generator.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    
    
summary(generator,(1,512,64,16))
model_state_dict = generator.state_dict()
new_state_dict = {k: v for k, v in model["state_dict"].items() if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

model_state_dict.update(new_state_dict)

generator.load_state_dict(model_state_dict)



mask=load_obj(mask_path)


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

for i,test_volume_path in enumerate(test_volume_paths[0:1]):
    try:
        print(test_volume_path.split('/')[-1]+'---------->'+str(i))
        original_volume=load_obj(test_volume_path)
        
    
        sub_sampled_volume=np.multiply(mask,original_volume).astype(np.uint8)
        
        ######## Normalize matrix###############################
        sub_sampled_volume_normalized= sub_sampled_volume/255
        
        bigger_reconstruction=reconstruct_volume_batches(sub_sampled_volume_normalized,generator,bigger_sub_volumes_dim)
        
        ######## Denormalize matrix###############################
        
        bigger_reconstruction=(bigger_reconstruction*127.5)+127.5
        bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
        
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
    except:
        print('Dimension ERROR...')
        

print('PSNR AVG: ',np.mean(PSNR_list))
print('RMSE AVG: ',np.mean(RMSE_list))
print('MAE AVG: ',np.mean(MAE_list))
print('SSIM AVG: ',np.mean(SSIM_list))
