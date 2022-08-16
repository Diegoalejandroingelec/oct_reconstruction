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
import os

def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()

bigger_sub_volumes_dim=(512,300,16)
original_volume_dim=(512,1000,100)
ngpu=2
masks_dataset_path=''
masks_dataset_path_train=''
results_dir='MODEL_EVALUATION_RANDOM_SUBSAMPLING_75'
model_path='BEST_MODEL_random_sampling.pth'
mask_path='../RANDOM_SAMPLING_DATASET/mask_random75.pkl'
txt_test_path='../RANDOM_GAUSSIAN_DATASET_SIGMA_150/test_volumes_paths.txt'

original_volumes_path='../../OCT_ORIGINAL_VOLUMES/'
comparison_size=100


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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

def reconstruct_volume_batches(volume,reconstruction_model,sub_volumes_dim):
    batch_size_for_inference=1
    batch_for_inference=[]
    batch_metadata_for_reconstruction=[]
    h_div_factor = sub_volumes_dim[0]
    w_div_factor = sub_volumes_dim[1]
    d_div_factor = sub_volumes_dim[2]
    reconstructed_volume = np.zeros(original_volume_dim)
    overlap_pixels_w=100
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
                            "w":(w_end+50,w_end+w_div_factor),
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

                            reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,50:,:]
                    
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
                print('here before')
                reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,:,:]
                print('here after')
            else:
                print('here2 before')
                reconstructed_volume[h_start:h_finish,w_start:w_finish,d_start:d_finish]=vol[:,50:,:]
                print('here2 after')
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




#mask_blue_noise=load_obj(mask_path)

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
    
    mean=np.mean(data)
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

def evaluate_model(mask_path,
                   txt_test_path,
                   original_volumes_path,
                   original_volume_dim,
                   reconstruction_model,
                   bigger_sub_volumes_dim,
                   comparison_size,
                   compare_with_roi=False):
    if(mask_path):
        mask=load_obj(mask_path)
    
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
            
            if(masks_dataset_path):
                f_gt = h5py.File(masks_dataset_path, 'r')
    
                vol_name=test_volume_path.split('/')[-1].split('.')[0]
    
                mask=np.array(f_gt.get(vol_name))
                if(mask.shape==()):
                    f_gt = h5py.File(masks_dataset_path_train, 'r')
        
                    vol_name=test_volume_path.split('/')[-1].split('.')[0]
        
                    mask=np.array(f_gt.get(vol_name))
        
            sub_sampled_volume=np.multiply(mask,original_volume).astype(np.uint8)
            
            ######## Normalize matrix###############################
            sub_sampled_volume_normalized,max_value=normalize(sub_sampled_volume)
            print(sub_sampled_volume_normalized.shape)
            bigger_reconstruction=reconstruct_volume_batches(sub_sampled_volume_normalized,reconstruction_model,bigger_sub_volumes_dim)
            print(bigger_reconstruction.shape)
            ######## Denormalize matrix###############################
            
            bigger_reconstruction=(bigger_reconstruction*max_value)+max_value
            bigger_reconstruction = bigger_reconstruction.astype(np.uint8)
            
            mask_blue_noise_prima= (~mask.astype(bool)).astype(int)
    
            bigger_reconstruction=np.multiply(mask_blue_noise_prima,bigger_reconstruction).astype(np.uint8)
            bigger_reconstruction=bigger_reconstruction+sub_sampled_volume
            
            
            if(compare_with_roi):
                print('ROI COMPARISON')
                window_for_comparison, upper_limit, lower_limit=get_window_for_comparison(original_volume,window_size=original_volume_dim,comparison_size=comparison_size)
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
                if(compare_with_roi):
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
            print('Dimension ERROR...')            
            print(e)

            
    
    print('PSNR AVG: ',np.mean(PSNR_list))
    print('RMSE AVG: ',np.mean(RMSE_list))
    print('MAE AVG: ',np.mean(MAE_list))
    print('SSIM AVG: ',np.mean(SSIM_list))
    
    
    if(compare_with_roi):
        print('Total PSNR AVG: ',np.mean(total_PSNR_list))
        print('Total RMSE AVG: ',np.mean(total_RMSE_list))
        print('Total MAE AVG: ',np.mean(total_MAE_list))
        print('Total SSIM AVG: ',np.mean(total_SSIM_list))


evaluate_model(mask_path,
               txt_test_path,
               original_volumes_path,
               original_volume_dim,
               reconstruction_model,
               bigger_sub_volumes_dim,
               comparison_size=100,
               compare_with_roi=True)

