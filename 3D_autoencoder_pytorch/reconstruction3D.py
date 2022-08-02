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
from Autoencoder_Architecture import Autoencoder

bigger_sub_volumes_dim=(512,150,16)
original_volume_dim=(512,1000,100)
ngpu=2
model_path='BEST_MODEL_random_sampling.pth'
mask_path='../RANDOM_SAMPLING_DATASET/mask_random75.pkl'
txt_test_path='../RANDOM_SAMPLING_DATASET/test_volumes_paths_random.txt'
original_volumes_path='../../OCT_ORIGINAL_VOLUMES/'
comparison_size=100
compare_with_roi=True

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

def get_window_for_comparison(original_volume,window_size,comparison_size):
    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)

    means=np.argmax(mean_b_scans,0)
    window_mean= np.mean(means).astype(int)
    window = np.zeros(window_size)
    upper_limit=window_mean-comparison_size if window_mean-comparison_size>=0 else 0
    lower_limit=window_mean+comparison_size if window_mean-comparison_size<=window_size[0] else window_size[0]
    window[upper_limit:lower_limit,:,:]=np.ones((lower_limit-upper_limit,window_size[1],window_size[2]))
    return window

def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()
      
def reconstruct_volume_batches(volume,reconstruction_model,sub_volumes_dim):
    batch_size_for_inference=1
    batch_for_inference=[]
    batch_metadata_for_reconstruction=[]
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
                            "w":(w_end+30,w_end+w_div_factor),
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
                            reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,30:,:]
                    
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

def initialize_reconstruction_model(model_path):
    # model_loaded= torch.load(model_path)
    # model_params = model_loaded.parameters()
    # param_list=[]
    # for weights in model_params:
    #     param_list.append(weights) 
    
    # global weights_index
    
    # weights_index=0
    # def weights_init(m):
    #     global weights_index
    #     classname = m.__class__.__name__  
    #     if(classname=='Conv3d' or classname=='ConvTranspose3d' or classname=='BatchNorm3d'):
    #         print(classname)
    #         weights_index+=2
    #         print(weights_index-2)
    #         m.weight=param_list[weights_index-2]
    #         print(weights_index-1)
    #         m.bias=param_list[weights_index-1]
    #     elif(classname=='PReLU'):
    #         print(classname)
    #         weights_index+=1
    #         print(weights_index-1)
    #         print(param_list[weights_index-1].shape)
    #         m.weight=param_list[weights_index-1]

    
    reconstruction_model = Autoencoder(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        reconstruction_model = nn.DataParallel(reconstruction_model, list(range(ngpu)))
    
    
    # # Apply the trained weights_init 
    # reconstruction_model.apply(weights_init)
    
    # summary(reconstruction_model, bigger_sub_volumes_dim)
    # return reconstruction_model
    
    # Load checkpoint model
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # Restore the parameters in the training node to this point
    
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = reconstruction_model.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    reconstruction_model.load_state_dict(model_state_dict)
    
    summary(reconstruction_model, bigger_sub_volumes_dim)
    return reconstruction_model


def evaluate_model(mask_path,
                   txt_test_path,
                   original_volumes_path,
                   original_volume_dim,
                   reconstruction_model,
                   bigger_sub_volumes_dim,
                   comparison_size,
                   compare_with_roi=False):
    
    mask=load_obj(mask_path)
    
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
            
        
            sub_sampled_volume=np.multiply(mask,original_volume).astype(np.uint8)
            
            ######## Normalize matrix###############################
            sub_sampled_volume_normalized,max_value=normalize(sub_sampled_volume)
            
            bigger_reconstruction=reconstruct_volume_batches(sub_sampled_volume_normalized,reconstruction_model,bigger_sub_volumes_dim)
            
            ######## Denormalize matrix###############################
            
            bigger_reconstruction=(bigger_reconstruction*max_value)+max_value
            bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
            
            mask_blue_noise_prima= (~mask.astype(bool)).astype(int)
    
            bigger_reconstruction=np.multiply(mask_blue_noise_prima,bigger_reconstruction).astype(np.uint8)
            bigger_reconstruction=bigger_reconstruction+sub_sampled_volume
            
            
            if(compare_with_roi):
                window_for_comparison=get_window_for_comparison(original_volume,window_size=original_volume_dim,comparison_size=comparison_size)
                volume_for_comparison=np.multiply(original_volume,window_for_comparison).astype(np.uint8)
                reconstruction_for_comparison=np.multiply(bigger_reconstruction,window_for_comparison).astype(np.uint8)
                
                PSNR=compute_PSNR(volume_for_comparison,reconstruction_for_comparison)
                RMSE=compute_RMSE(volume_for_comparison,reconstruction_for_comparison)
                MAE=compute_MAE(volume_for_comparison,reconstruction_for_comparison)
                SSIM=ssim(volume_for_comparison.astype(np.uint8),reconstruction_for_comparison.astype(np.uint8))
            else:
                PSNR=compute_PSNR(original_volume,bigger_reconstruction)
                RMSE=compute_RMSE(original_volume,bigger_reconstruction)
                MAE=compute_MAE(original_volume,bigger_reconstruction)
                SSIM=ssim(original_volume.astype(np.uint8),bigger_reconstruction.astype(np.uint8))
            
            PSNR_list.append(PSNR)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            SSIM_list.append(SSIM)
            
            if(i%10==0):
                if(compare_with_roi):
                    print('Saving images... ')
                    cv2.imwrite('REFERENCE_VOLUME_SLICE_'+str(i)+'.jpeg', volume_for_comparison[:,:,50])
                    cv2.imwrite('RECONSTRUCTION_VOLUME_SLICE_'+str(i)+'.jpeg', reconstruction_for_comparison[:,:,50])
                    cv2.imwrite('SUBSAMPLED_VOLUME_SLICE_'+str(i)+'.jpeg', sub_sampled_volume[:,:,50])
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

reconstruction_model=initialize_reconstruction_model(model_path)
evaluate_model(mask_path,
               txt_test_path,
               original_volumes_path,
               original_volume_dim,
               reconstruction_model,
               bigger_sub_volumes_dim,
               comparison_size,
               compare_with_roi)



