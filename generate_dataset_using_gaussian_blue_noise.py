#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:04:24 2022

@author: diego
"""

from scipy.io import loadmat
from glob import glob
import numpy as np
import random
import pickle
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
import random

vol_dims=(512,1000,100)



#'blue_noise_subsampling'
#'raster_subsampling'
#'random_subsampling'

subsampling_method='random_subsampling'

def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume


########################################################################################
#######################
#######################
#######################                   UTILS
#######################
#######################
########################################################################################
def save_obj(obj,path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
    
########################################################################################
#######################
#######################
#######################                   EXTRACT SUB VOLUMES FOR TRAINING AND TESTING
#######################
#######################
########################################################################################

def extract_sub_volumes(volume,name,h5_file):
    w_div_factor = 64
    h_div_factor = 512
    d_div_factor = 16
    
    overlap_pixels_w=w_div_factor//2
    overlap_pixels_h=0
    overlap_pixels_d=d_div_factor//2
    
    index=0
    d_end=0
    can_iterate_over_d=True
    
    while can_iterate_over_d:
    #for d in range(int(np.ceil(volume.shape[2]/d_div_factor))):
        w_end=0
        can_iterate_over_w=True
        while can_iterate_over_w:
        #for w in range(int(np.ceil(volume.shape[1]/w_div_factor))):
            h_end=0
            can_iterate_over_h=True
            while can_iterate_over_h:
            #for h in range(int(np.ceil(volume.shape[0]/h_div_factor))):
                
                # print('heigh: ',(h_end,h_end+h_div_factor))
                # print('width: ',(w_end,w_end+w_div_factor))
                # print('depth: ',(d_end,d_end+d_div_factor))
                sub_volume=volume[h_end:h_end+h_div_factor,w_end:w_end+w_div_factor,d_end:d_end+d_div_factor]
                if(sub_volume.shape!=(512,64,16)):
                    raise Exception("ERROR GENERATING SUB VOLUMES")
                    
                    
                h5_file.create_dataset(name+'_'+str(index), data=sub_volume)
                #save_obj(sub_volume,name+'_'+str(index))
                index+=1
                
                h_end=h_end+h_div_factor-overlap_pixels_h
                if(h_end+h_div_factor>=volume.shape[0]):
                    h_end=volume.shape[0]-h_div_factor
                    can_iterate_over_h=False
                
            w_end=w_end+w_div_factor-overlap_pixels_w
            if(w_end+w_div_factor>=volume.shape[1]):
                w_end=volume.shape[1]-w_div_factor
                can_iterate_over_w=False
                
        d_end=d_end+d_div_factor-overlap_pixels_d
        if(d_end+d_div_factor>=volume.shape[2]):
            d_end=volume.shape[2]-d_div_factor
            can_iterate_over_d=False
         
    # print(index)
    
    
########################################################################################
#######################
#######################
#######################                   GET VOLUME PATHS
#######################
#######################
########################################################################################   


def get_volume_paths():
    amd_eyes_paths = glob("../oct_original_volumes/AMD/*.mat", recursive = True)
    control_eyes_paths= glob("../oct_original_volumes/Control/*.mat", recursive = True)
    all_paths = amd_eyes_paths+control_eyes_paths
    random.shuffle(all_paths)
    return all_paths

########################################################################################
#######################
#######################
#######################                   GAUSSIAN BLUE NOISE MASK
#######################
#######################
########################################################################################  


def generate_gaussian_blue_noise_mask(original_volume,desired_transmittance,sigma,plot_mask):
    def gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
        
    def create_blue_noise_mask_1(expected_dims,subsampling_percentage):
        blue_noise_cube1 = np.transpose(np.load('blue_noise_cubes/bluenoisecube.npy'), (2,1,0))
        blue_noise_cube2 = np.transpose(np.load('blue_noise_cubes/bluenoisecube2.npy'), (2,1,0))
        blue_noise_cube3 = np.transpose(np.load('blue_noise_cubes/bluenoisecube3.npy'), (2,1,0))
        blue_noise_cube4 = np.transpose(np.load('blue_noise_cubes/bluenoisecube4.npy'), (2,1,0))
        
        blue_noise_cube5 = np.transpose(np.load('blue_noise_cubes/bluenoisecube5.npy'), (2,1,0))
        blue_noise_cube6 = np.transpose(np.load('blue_noise_cubes/bluenoisecube6.npy'), (2,1,0))
        blue_noise_cube7 = np.transpose(np.load('blue_noise_cubes/bluenoisecube7.npy'), (2,1,0))
        blue_noise_cube8 = np.transpose(np.load('blue_noise_cubes/bluenoisecube8.npy'), (2,1,0))
        
        concat1=np.concatenate((blue_noise_cube1,blue_noise_cube2),axis=0)
        concat2=np.concatenate((blue_noise_cube3,blue_noise_cube4),axis=0)
        
        concat3=np.concatenate((blue_noise_cube5,blue_noise_cube6),axis=0)
        concat4=np.concatenate((blue_noise_cube7,blue_noise_cube8),axis=0)
        
        concat5=np.concatenate((concat1,concat2,concat3,concat4),axis=1)
        
        blue_noise_mask = concat5[0:expected_dims[0],0:expected_dims[1],0:expected_dims[2]]
        
        blue_noise_mask=blue_noise_mask/np.max(blue_noise_mask)
        
        binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
        binary_blue_noise_mask = binary_blue_noise_mask*1
        total = binary_blue_noise_mask.sum()
        
        
        missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
        print('Blue noise missing data: ', missing_data)
        
        return blue_noise_mask,binary_blue_noise_mask
        
     


    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)

    means=np.argmax(mean_b_scans,0)
    #means_smooth=savgol_filter(means,51,1)


    # plt.imshow(mean_b_scans,cmap='gray')
    # plt.plot(means_smooth)
    # plt.show()


    gaussian_mask=np.ones((512,1000))

    for i in range(vol_dims[1]):
        for j in range(vol_dims[0]):
            likelihood=gaussian(j, 1, means[i], sigma)
            threshold= random.uniform(0, 1)
            if(threshold>likelihood):
               gaussian_mask[j,i]=0
        
      
    gaussian_mask = np.repeat(gaussian_mask[None,:], vol_dims[2], axis=0)
    gaussian_mask=np.transpose(gaussian_mask,(1,2,0))


    # plt.imshow(gaussian_mask[:,:,11],cmap='gray')
    # plt.show() 


    gaussian_mask_transmittance=(gaussian_mask.sum())/(vol_dims[0]*vol_dims[1]*vol_dims[2])
    # print(gaussian_mask_transmittance)


    desired_transmittance=0.25

    blue_noise_transmitance=desired_transmittance/gaussian_mask_transmittance

    _,blue_noise_mask=create_blue_noise_mask_1(vol_dims,subsampling_percentage=1-blue_noise_transmitance)



    mask = np.multiply(gaussian_mask.astype(np.uint8),blue_noise_mask.astype(np.uint8))

    print('Missing Data: ', 100-(mask.sum()*100)/(vol_dims[0]*vol_dims[1]*vol_dims[2]))
    if(plot_mask):
        plt.imshow(mask[:,:,11],cmap='gray')
        plt.show() 
    return mask
########################################################################################
#######################
#######################
#######################                   GENERATE DATASET
#######################
#######################
########################################################################################

def generate_dataset():
    all_paths=get_volume_paths()
    
    training_total= int(np.floor(len(all_paths)*0.8))
    
    train_volumes_paths=all_paths[0:training_total]
    test_volumes_paths=all_paths[training_total:]
    
    
    volume_number=0
    subsampled_volumes_dataset_train = h5py.File('training_random_subsampled_volumes.h5', 'w')
    volumes_dataset_train = h5py.File('training_random_ground_truth.h5', 'w')
    masks_dataset_train = h5py.File('masks_dataset_train.h5', 'w')
    with open('train_volumes_paths_random.txt', 'w') as f:
        f.write('\n'.join(train_volumes_paths))
        
    for volume_path in train_volumes_paths:

        print(volume_path)
        try: 
            volume = read_data(volume_path)
            mask=generate_gaussian_blue_noise_mask(original_volume=volume,desired_transmittance=0.25,sigma=51,plot_mask=True)
            name=volume_path.split('/')[-1].split('.')[0]
            masks_dataset_train.create_dataset(name, data=mask)
            subsampled_image = np.multiply(mask,volume).astype(np.uint8)
            
            
            name='original_train_vol_'+str(volume_number)
            extract_sub_volumes(volume,name,volumes_dataset_train)
                
            name='subsampled_train_vol_'+str(volume_number)
            extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_train)
            
            volume_number+=1
        
        except:
            print('WRONG dimentions'+volume_path)

    
    
    subsampled_volumes_dataset_train.close()  
    volumes_dataset_train.close()
    masks_dataset_train.close()
    
    
    
    volume_number=0
    subsampled_volumes_dataset_test = h5py.File('testing_random_subsampled_volumes.h5', 'w')
    volumes_dataset_test = h5py.File('testing_random_ground_truth.h5', 'w')
    masks_dataset_test = h5py.File('masks_dataset_test.h5', 'w')
    with open('test_volumes_paths_random.txt', 'w') as f:
        f.write('\n'.join(test_volumes_paths))
        
        
    for volume_path in test_volumes_paths:
            
            try:
                print(volume_path)
        
                volume = read_data(volume_path)
                mask=generate_gaussian_blue_noise_mask(original_volume=volume,desired_transmittance=0.25,sigma=51,plot_mask=True)
                name=volume_path.split('/')[-1].split('.')[0]
                masks_dataset_test.create_dataset(name, data=mask)
                subsampled_image = np.multiply(mask,volume).astype(np.uint8)
                
                
                name='original_test_vol_'+str(volume_number)
                extract_sub_volumes(volume,name,volumes_dataset_test)
                
                name='subsampled_test_vol_'+str(volume_number)
                extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_test)
                
                volume_number+=1
            
            except:
                print('WRONG dimentions'+volume_path)
            

            
            
    subsampled_volumes_dataset_test.close()  
    volumes_dataset_test.close()
    masks_dataset_test.close()
    
generate_dataset()  