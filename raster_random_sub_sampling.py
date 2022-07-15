#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:30:43 2022

@author: diego
"""
from scipy.io import loadmat
from glob import glob
import numpy as np
# import cv2
import random
import pickle
import h5py

sub_sampling_percentage=75

def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume



def generate_mask(percentage,volume_x,volume_y,volume_z):
    volume_dim= (volume_x,volume_y,volume_z)
    
    percentage_of_missing_data=percentage
    
    number_of_columns_z_axis= int(np.ceil((volume_z*percentage_of_missing_data)/2))
    
    number_of_columns_x_axis= int(np.ceil((volume_x*(percentage_of_missing_data*volume_z-number_of_columns_z_axis))/volume_z))
    
    
    columns_to_be_deleted_z_axis = []
    columns_to_be_deleted_x_axis = []
    
    while len(columns_to_be_deleted_z_axis)  != number_of_columns_z_axis:
        column = random.randint(0, volume_dim[2]-1)
        if column not in columns_to_be_deleted_z_axis:
            columns_to_be_deleted_z_axis.append(column)
            
            
    #columns_to_be_deleted_z_axis.sort() 
       
    while len(columns_to_be_deleted_x_axis) != number_of_columns_x_axis:
        column = random.randint(0, volume_dim[0]-1)
        if column not in columns_to_be_deleted_x_axis:
            columns_to_be_deleted_x_axis.append(column)
            
    #columns_to_be_deleted_x_axis.sort()
    
     
    volume_mask = []
    for i in range(volume_dim[0]):
        if i in columns_to_be_deleted_x_axis:
            a_direction = np.zeros((volume_dim[1],1),dtype='uint8')
        else:
            a_direction = np.ones((volume_dim[1],1),dtype='uint8')
        try:
          b_mask = np.concatenate((b_mask,a_direction),axis=1)
        except:
          b_mask = a_direction
          
    for i in range(volume_dim[2]):
        if i in columns_to_be_deleted_z_axis:
            volume_mask.append(np.zeros((volume_dim[1],volume_dim[0])))
        else:
            volume_mask.append(b_mask) 
            
    volume_mask=np.array(volume_mask,dtype='uint8')
            
    volume_mask= np.transpose(volume_mask,(1,2,0)) 
        
    return volume_mask

def save_obj(obj,path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

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
    
mask = generate_mask(percentage=sub_sampling_percentage/100,volume_x=1000,volume_y=512,volume_z=100)
save_obj(mask,'/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/masks/mask_GAN_'+str(sub_sampling_percentage))

def get_volume_paths():
    amd_eyes_paths = glob("/home/diego/Downloads/oct_images/AMD/*.mat", recursive = True)
    control_eyes_paths= glob("/home/diego/Downloads/oct_images/Control/*.mat", recursive = True)
    all_paths = amd_eyes_paths+control_eyes_paths
    random.shuffle(all_paths)
    return all_paths


def generate_dataset(mask):
    all_paths=get_volume_paths()
    
    training_total= int(np.floor(len(all_paths)*0.8))
    
    train_volumes_paths=all_paths[0:training_total]
    test_volumes_paths=all_paths[training_total:]
    
    
    volume_number=0
    subsampled_volumes_dataset_train = h5py.File('training_subsampled_volumes.h5', 'w')
    volumes_dataset_train = h5py.File('training_ground_truth.h5', 'w')
    for volume_path in train_volumes_paths:

        print(volume_path)
        try: 
            volume = read_data(volume_path)
        
            subsampled_image = np.multiply(mask,volume)

            
            name='original_train_vol_'+str(volume_number)
            extract_sub_volumes(volume,name,volumes_dataset_train)
                
            name='subsampled_train_vol_'+str(volume_number)
            extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_train)
            
            volume_number+=1
        
        except:
            print('WRONG dimentions'+volume_path)

    
    
    subsampled_volumes_dataset_train.close()  
    volumes_dataset_train.close()
    
    
    
    
    volume_number=0
    subsampled_volumes_dataset_test = h5py.File('testing_subsampled_volumes.h5', 'w')
    volumes_dataset_test = h5py.File('testing_ground_truth.h5', 'w')
    for volume_path in test_volumes_paths:
            
            try:
                print(volume_path)
        
                volume = read_data(volume_path)
                
                subsampled_image = np.multiply(mask,volume)
                
                
                name='original_test_vol_'+str(volume_number)
                extract_sub_volumes(volume,name,volumes_dataset_test)
                
                name='subsampled_test_vol_'+str(volume_number)
                extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_test)
                
                volume_number+=1
            
            except:
                print('WRONG dimentions'+volume_path)
            

            
            
    subsampled_volumes_dataset_test.close()  
    volumes_dataset_test.close()
        
generate_dataset(mask)   

# all_paths=get_volume_paths()
# for volume_path in all_paths:
#     try:
#         volume = read_data(volume_path)
#         subsampled_image = np.multiply(mask,volume)
#         save_obj(subsampled_image,'/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/subsampled_'+str(sub_sampling_percentage)+'/subsapled_'+volume_path.split('/')[-1].split('.')[0])
#     except:
#         print('WRONG dimentions'+volume_path)
        
        
        
# subsampled_image=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/subsampled_23/subsapled_Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')
# original_volume =load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')


# for i in range(100):
#     cv2.imshow('Original Image',original_volume[:,:,i])
#     cv2.imshow('Sumbsampled Image',subsampled_image[:,:,i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


    
