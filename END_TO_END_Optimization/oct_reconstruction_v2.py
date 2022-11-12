#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:29:35 2022

@author: diego
"""

import os
import h5py
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
from architectures import Autoencoder,Risley_Speeds
import bm3d
from skimage import img_as_float
import time
from scipy.io import loadmat
from risley_varying_all_parameters import create_risley_pattern 


sub_vol=(512,200,16)
bigger_sub_volumes_dim=(512,200,16)
original_volume_dim=(512,1000,100)
ngpu=2
denoised_dataset_folder_path='../DATASET_DENOISED'
results_dir='MODEL_EVALUATION_2'

model_path='END_TO_END_OPTIMIZATION/BEST_MODEL_autoencoder_0.pth.tar'
speeds_model_path='END_TO_END_OPTIMIZATION/BEST_MODEL_speeds_epoch_0.pth.tar'


txt_test_path='../3D_autoencoder_pytorch/fast_test_paths.txt'
original_volumes_path='../../OCT_ORIGINAL_VOLUMES/'

comparison_size=100
compare_with_roi=True
denoised_ground_truth_for_comparison=True
reconstruct_with_motion=False
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def create_3D_mask(w1,w2,w3,w4,original_volume=None,create_with_motion=False):
    
    expected_dims=original_volume_dim
    band_width=176
    line_width=band_width/expected_dims[0]
    start_wavelength=962 


    
    mask_risley=create_risley_pattern(w1,
                              w2,
                              w3,
                              w4,
                              expected_dims,
                              line_width,
                              start_wavelength,
                              original_volume,
                              x_translation=500,
                              y_translation=50,
                              x_factor_addition=9.5,
                              y_factor_addition=1.5,
                              tf=12,
                              PRF=8000000,#2500000,
                              a=10*(np.pi/180),
                              number_of_prisms=4,
                              maximum_transmittance=0.43,
                              minimum_transmittance=0.0,
                              sigma=150,
                              transmittance_distribution_fn='ga',
                              number_of_laser_sweeps=250,
                              steps_before_centering=10,
                              hand_tremor_period=1/9,
                              laser_time_between_sweeps=7.314285714285714e-05,
                              x_factor=50,
                              y_factor=50,
                              generate_volume_with_motion=create_with_motion,
                              apply_motion=create_with_motion,
                              plot_mask=False)
    if(create_with_motion):
        return mask_risley[0],mask_risley[1]
    else:
        return mask_risley[0],mask_risley[1]

def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume

def find_denoised_volume(volume_path,denoised_dataset_folder_path):
    volume_name=volume_path.split('/')[-1].split('.')[0]+'.mat'
    path=denoised_dataset_folder_path+'/denoised_'+volume_name
    return read_data(path)

def median_filter_3D(volume,threshold,window_size):
    start = time.process_time()
    filtered_volume=np.zeros(volume.shape)
    volume=np.transpose(volume,(2,0,1))
    print("DENOISING VOLUME...")
    for index,image in enumerate(volume):
        padding_size=window_size//2
        img_with_padding=cv2.copyMakeBorder(image,
                                            padding_size,
                                            padding_size,
                                            padding_size,
                                            padding_size,
                                            cv2.BORDER_REPLICATE)
        filtered_image=np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                central_value=image[i,j]
                if(central_value<threshold):
                    values=img_with_padding[i:i+window_size,j:j+window_size]
                    median=np.median(values)
                    filtered_image[i,j]=median
                else:
                    filtered_image[i,j]=central_value
        filtered_volume[:,:,index]=filtered_image
    print('TIME ELAPSED FOR DENOISING VOLUME:', time.process_time() - start, 's')
    return filtered_volume.astype(np.uint8)

def BM3D_denoiser(volume,sigma_psd=0.1):
    print('Denoising volume...')
    start = time.process_time()
    noisy_img=  img_as_float(volume)
    BM3D_denoised=bm3d.bm3d(noisy_img,
                            sigma_psd,
                            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
    print('TIME ELAPSED FOR DENOISING VOLUME:', time.process_time() - start, 's')
    return (BM3D_denoised*255).astype(np.uint8)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

def get_window_for_comparison(original_volume,window_size,comparison_size):
    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)

    means=np.argmax(mean_b_scans,0)+30
    window_mean= np.mean(means).astype(int)
    window = np.zeros(window_size)
    upper_limit=window_mean-comparison_size if window_mean-comparison_size>=0 else 0
    lower_limit=window_mean+comparison_size if window_mean-comparison_size<=window_size[0] else window_size[0]
    window[upper_limit:lower_limit,:,:]=np.ones((lower_limit-upper_limit,window_size[1],window_size[2]))
    return window,upper_limit,lower_limit

def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()

def predict_best_angular_speeds(volume,sub_vol,speeds_model):
    batch_size_for_inference=4
    batch_for_inference=[]
    h_div_factor = sub_vol[0]
    w_div_factor = sub_vol[1]
    d_div_factor = sub_vol[2]
    overlap_pixels_w=0
    d_end=0
    angular_speeds=[]
    for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        can_iterate_over_w=True
        last_iteration_w = False
        while can_iterate_over_w:
            h_end=0
            for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
        
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                sub_volume_normalized,_=normalize(sub_volume)
                batch_for_inference.append(sub_volume_normalized)
       

                    

                if len(batch_for_inference)==batch_size_for_inference:
                    batch_for_inference=np.array(batch_for_inference)
                    batch_for_inference=torch.from_numpy(batch_for_inference).to(device, dtype=torch.float)
                    angular_speeds.append(speeds_model(batch_for_inference).cpu().detach().numpy())
                    batch_for_inference=[]

                h_end=h_end+h_div_factor
                if(h_end+h_div_factor>volume.shape[0]):
                    h_end=volume.shape[0]-h_div_factor
                    
            w_end=w_end+w_div_factor-overlap_pixels_w
            if(last_iteration_w):
                can_iterate_over_w=False
            if(w_end+w_div_factor>volume.shape[1]):
                w_end=volume.shape[1]-w_div_factor
                last_iteration_w = True

                
        d_end=d_end+d_div_factor
        if(d_end+d_div_factor>volume.shape[2]):
            d_end=volume.shape[2]-d_div_factor
            
    if len(batch_for_inference)>0:
        batch_for_inference=np.array(batch_for_inference)
       
        batch_for_inference=torch.from_numpy(batch_for_inference).to(device, dtype=torch.float)
        
        angular_speeds.append(speeds_model(batch_for_inference).cpu().detach().numpy())
        
        
       
              
                
    angular_speeds=np.concatenate(angular_speeds)            
    return np.mean(angular_speeds,0)




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
                reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,30:,:]
              
                
                
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

def initialize_speeds_generator_model(model_path):
    speeds_generator_model = Risley_Speeds(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        speeds_generator_model = nn.DataParallel(speeds_generator_model, list(range(ngpu)))
        
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_state_dict = speeds_generator_model.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    speeds_generator_model.load_state_dict(model_state_dict)
    
    summary(speeds_generator_model, sub_vol)
    return speeds_generator_model
    
    
def mark_ROI(upper_limit,lower_limit,name,img):
    start_point_u = (0, upper_limit)
    end_point_u=(1000, upper_limit)

    start_point_l = (0, lower_limit)
    end_point_l=(1000, lower_limit)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # Green color in BGR
    color = (0, 0, 255)
     
    # Line thickness of 9 px
    thickness = 2
     
    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 9 px
    image = cv2.line(img, start_point_u, end_point_u, color, thickness)
    image = cv2.line(image, start_point_l, end_point_l, color, thickness)
    cv2.imwrite(name+'.jpeg', image)

def plot_error(data,name):
    fig, ax1 = plt.subplots(1,1,figsize=(10, 5))
    mean=18.77482421875
    print('MEAN OF ERROR ',mean)
    im = ax1.imshow(data, vmin=0, vmax=mean+50)
    cax = fig.add_axes([ax1.get_position().x1+0.01,
                        ax1.get_position().y0,0.02,
                        ax1.get_position().height])
    cb = plt.colorbar(im,cax=cax,extend="max")
    
    fg_color = 'black'
    bg_color = 'white'
    
    # IMSHOW    
    # set title plus title color
    ax1.set_title('Reconstruction Error', color=fg_color)
    
    # set figure facecolor
    ax1.patch.set_facecolor(bg_color)
    
    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)
    
    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)    
    
    # COLORBAR
    # set colorbar label plus label color
    cb.set_label('Absolute Error', color=fg_color)
    
    # set colorbar tick color
    cb.ax.yaxis.set_tick_params(color=fg_color)
    
    # set colorbar edgecolor 
    cb.outline.set_edgecolor(fg_color)
    
    
    
    fig.patch.set_facecolor(bg_color)    
    
    plt.show()
    fig.savefig(name, dpi=200, facecolor=bg_color)

def create_mask_spectrum(mask):
    pattern=mask[250,:,:]-np.mean(mask[250,:,:])
    pattern=pattern.T
    DFT=np.fft.fftshift(np.fft.fft2(pattern))
    fontsize=60
    fontsizet=80
    fig, (ax2, ax1) = plt.subplots(2,1)
    fig.tight_layout(pad=-2.5)
    fig.set_size_inches(50, 40)
    ax2.imshow(pattern*255,cmap='gray')
    ax2.set_title('Risley Beam Steering Pattern Optimized',fontsize=fontsizet)
    im = ax1.imshow(20*np.log(np.abs(DFT)),
                cmap="viridis",
                interpolation="nearest",
                vmin=0.0,
                vmax=np.percentile(20*np.log(np.abs(DFT)),99))
    ax1.set_title('Magnitude Spectrum',fontsize=fontsizet)
    ax2.axis('off')
    ax1.axis('off')
    cax = fig.add_axes([ax1.get_position().x1+0.01,
                        ax1.get_position().y0,0.02,
                        ax1.get_position().height])
    cb = plt.colorbar(im,cax=cax,extend="max")
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel('Magnitude (dB)',fontsize=fontsize)
    fig.savefig('mask spectrum', dpi=400, facecolor='white')

def visualize_3D_spectrum(mask):
    import napari
    img=mask*255
    DFT=np.fft.fftshift(np.fft.fftn(img))/float(np.size(img));
    magnitude=20*np.log(np.abs(DFT))
    viewer = napari.view_image(magnitude)
    
def evaluate_model(denoised_dataset_folder_path,
                   txt_test_path,
                   original_volumes_path,
                   original_volume_dim,
                   reconstruction_model,
                   speeds_generator,
                   bigger_sub_volumes_dim,
                   comparison_size,
                   reconstruct_with_motion=False,
                   compare_with_roi=False):
    
    
    with open(txt_test_path) as f:
        lines = f.readlines()
    test_volume_paths=[ original_volumes_path+name.split('/')[-1].split('.')[0]+'.pkl' for name in lines]    
    
    
    PSNR_list=[]
    RMSE_list=[]
    MAE_list=[]
    SSIM_list=[]
    
    total_PSNR_list=[]
    total_RMSE_list=[]
    total_MAE_list=[]
    total_SSIM_list=[]
    
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    for i,test_volume_path in enumerate(test_volume_paths):
        try:
            print(test_volume_path.split('/')[-1]+'---------->'+str(i))
            original_volume=load_obj(test_volume_path)
            
            #speeds_pred=speeds_generator(torch.tensor(np.expand_dims(original_volume[0:sub_vol[0],0:sub_vol[1],0:sub_vol[2]],axis=0)).to(device,dtype=torch.float)).cpu().detach().numpy()

            speeds_pred=predict_best_angular_speeds(original_volume,
                                                    sub_vol,
                                                    speeds_generator)
            
            # speeds_pred=speeds_generator(torch.tensor(original_volume).to(device,dtype=torch.float)).cpu().detach().numpy()
            print(speeds_pred*100000)
            if (reconstruct_with_motion):
                mask,sub_sampled_volume=create_3D_mask(w1=speeds_pred[0]*100000,
                                w2=speeds_pred[1]*100000,
                                w3=speeds_pred[2]*100000,
                                w4=speeds_pred[3]*100000,
                                original_volume=original_volume,
                                create_with_motion=True)
            else:
                mask,transmittance=create_3D_mask(w1=speeds_pred[0]*100000,
                                w2=speeds_pred[1]*100000,
                                w3=speeds_pred[2]*100000,
                                w4=speeds_pred[3]*100000,
                                original_volume=None)
            
                sub_sampled_volume=np.multiply(mask,original_volume).astype(np.uint8)
            
            #create_mask_spectrum(mask)
            #visualize_3D_spectrum(mask)
            
            ####### Normalize matrix###############################
            sub_sampled_volume_normalized,max_value=normalize(sub_sampled_volume)
            print(sub_sampled_volume_normalized.shape)
            bigger_reconstruction=reconstruct_volume_batches(sub_sampled_volume_normalized,
                                                             reconstruction_model,
                                                             bigger_sub_volumes_dim)
            print(bigger_reconstruction.shape)
            ######## Denormalize matrix###############################
            max_value=127.5
            bigger_reconstruction=(bigger_reconstruction*max_value)+max_value
            bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
            
            mask_blue_noise_prima= (~mask.astype(bool)).astype(int)
    
            bigger_reconstruction=np.multiply(mask_blue_noise_prima,bigger_reconstruction).astype(np.uint8)
            if(denoised_ground_truth_for_comparison):
                denoised_original_volume=find_denoised_volume(test_volume_path,denoised_dataset_folder_path)
                #denoised_original_volume=median_filter_3D(original_volume,40,5)
                sub_sampled_denoised_original_volume=np.multiply(mask,denoised_original_volume).astype(np.uint8)
                bigger_reconstruction=bigger_reconstruction+sub_sampled_denoised_original_volume
            else:
                bigger_reconstruction=bigger_reconstruction+sub_sampled_volume
            
            
            if(compare_with_roi):
                print('ROI COMPARISON')
                window_for_comparison, upper_limit, lower_limit=get_window_for_comparison(original_volume,window_size=original_volume_dim,comparison_size=comparison_size)
                if(denoised_ground_truth_for_comparison):
                    volume_for_comparison=np.multiply(denoised_original_volume,window_for_comparison).astype(np.uint8)
                    original_volume=denoised_original_volume
                else:
                    volume_for_comparison=np.multiply(original_volume,window_for_comparison).astype(np.uint8)
                
                reconstruction_for_comparison=np.multiply(bigger_reconstruction,window_for_comparison).astype(np.uint8)
                print('evaluate metrics on roi')
                PSNR=compute_PSNR(volume_for_comparison,reconstruction_for_comparison)
                RMSE=compute_RMSE(volume_for_comparison,reconstruction_for_comparison)
                MAE=compute_MAE(volume_for_comparison,reconstruction_for_comparison)
                SSIM=ssim(volume_for_comparison.astype(np.uint8),reconstruction_for_comparison.astype(np.uint8))
                print('evaluate metrics on the whole volume')
                
                total_PSNR=compute_PSNR(original_volume,bigger_reconstruction)
                total_RMSE=compute_RMSE(original_volume,bigger_reconstruction)
                total_MAE=compute_MAE(original_volume,bigger_reconstruction)
                total_SSIM=ssim(original_volume.astype(np.uint8),bigger_reconstruction.astype(np.uint8))
            else:
                PSNR=compute_PSNR(original_volume,bigger_reconstruction)
                RMSE=compute_RMSE(original_volume,bigger_reconstruction)
                MAE=compute_MAE(original_volume,bigger_reconstruction)
                SSIM=ssim(original_volume.astype(np.uint8),bigger_reconstruction.astype(np.uint8))
            
            PSNR_list.append(PSNR)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            SSIM_list.append(SSIM)
            
            if(compare_with_roi):
                total_PSNR_list.append(total_PSNR)
                total_RMSE_list.append(total_RMSE)
                total_MAE_list.append(total_MAE)
                total_SSIM_list.append(total_SSIM)
            
            if(i%10==0):
                print('PSNR AVG: ',np.mean(PSNR_list))
                print('RMSE AVG: ',np.mean(RMSE_list))
                print('MAE AVG: ',np.mean(MAE_list))
                print('SSIM AVG: ',np.mean(SSIM_list))
    
    

                if(compare_with_roi):
                    print('Total PSNR AVG: ',np.mean(total_PSNR_list))
                    print('Total RMSE AVG: ',np.mean(total_RMSE_list))
                    print('Total MAE AVG: ',np.mean(total_MAE_list))
                    print('Total SSIM AVG: ',np.mean(total_SSIM_list)) 
                    print('Saving images... ')
                    
                    mark_ROI(upper_limit,lower_limit,
                             os.path.join(results_dir, f"REFERENCE_VOLUME_SLICE_{i}"),
                             original_volume[:,:,50])
                    
                    mark_ROI(upper_limit,lower_limit,
                             os.path.join(results_dir, f"RECONSTRUCTION_VOLUME_SLICE_{i}"),
                             bigger_reconstruction[:,:,50])
                    
                    mark_ROI(upper_limit,
                             lower_limit,
                             os.path.join(results_dir, f"SUBSAMPLED_VOLUME_SLICE_{i}"),
                             sub_sampled_volume[:,:,50])
                    
                    error=np.abs(original_volume.astype(np.float32)-bigger_reconstruction.astype(np.float32))
                    error=error.astype(np.uint8)
                    plot_error(data=error[:,:,50],
                               name=os.path.join(results_dir, f'RECONSTRUCTION_ERROR_{i}.png'))
                    plt.imshow(error[:,:,50])
                    plt.show()
                print('Generating video...')
                gap=np.zeros((512,50,100)).astype(np.uint8)
                comparative_volume=np.concatenate((original_volume,gap,bigger_reconstruction,gap,sub_sampled_volume),axis=1)
                make_video(comparative_volume,
                           os.path.join(results_dir, f'comparative_reconstruction_{i}'))
        except Exception as e:
            print(e)
            print('Dimension ERROR...')
            
    
    print('PSNR AVG: ',np.mean(PSNR_list))
    print('RMSE AVG: ',np.mean(RMSE_list))
    print('MAE AVG: ',np.mean(MAE_list))
    print('SSIM AVG: ',np.mean(SSIM_list))
    
    
    if(compare_with_roi):
        print('Total PSNR AVG: ',np.mean(total_PSNR_list))
        print('Total RMSE AVG: ',np.mean(total_RMSE_list))
        print('Total MAE AVG: ',np.mean(total_MAE_list))
        print('Total SSIM AVG: ',np.mean(total_SSIM_list))

#TODO : FIX UPLOAD WEIGHTS

reconstruction_model=initialize_reconstruction_model(model_path)

speeds_generator_model=initialize_speeds_generator_model(speeds_model_path)


evaluate_model(denoised_dataset_folder_path,
               txt_test_path,
               original_volumes_path,
               original_volume_dim,
               reconstruction_model,
               speeds_generator_model,
               bigger_sub_volumes_dim,
               comparison_size,
               reconstruct_with_motion,
               compare_with_roi)