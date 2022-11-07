#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:30:43 2022

@author: diego
"""
import os
from risley_beam_steering import get_risley_3D_mask
from Generate_shifting_blue_noise import generate_shifting_blue_noise,generate_binary_blue_noise_mask
from scipy.io import loadmat
from glob import glob
import numpy as np
import random
import pickle
import h5py
import matplotlib.pyplot as plt
from risley_beam_steering_real_pattern import create_risley_pattern,required_prf
import sys
sys.path.insert(0, '/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/EyeTrackingData')

#from data_evaluation import add_motion_to_volume
#from data_evaluation import add_motion_to_volume



sub_sampling_percentage=100

#'blue_noise_subsampling'
#'raster_subsampling'
#'random_subsampling'
#'risley_subsampling'
subsampling_method='raster_subsampling'

def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume

def find_denoised_volume(volume_path,denoised_dataset_folder_path):
    volume_name=volume_path.split('/')[-1]
    path=denoised_dataset_folder_path+'/denoised_'+volume_name
    return read_data(path)
########################################################################################
#######################
#######################
#######################                   RISLEY BEAM STEERING 4 PRISMS
#######################
#######################
########################################################################################

def create_risley_beam_steering_mask_4_prisms(expected_dims,desired_transmittance):

    number_of_prisms=4

    #Laser Pulse Rate
    PRF=required_prf(desired_transmittance)#1999000
    #PRF=None
    #Image Capture Time 0.003
    tf=0.016

    #angular speed risley 1 rotations per sec
    w=9990
    #angula speed risley 2 rotations per sec
    w2=111000

    #angula speed risley 2 rotations per sec
    w3=12333

    #angula speed risley 2 rotations per sec
    w4=119538

    a=10*(np.pi/180)    
    expected_dims=(512,1000,100)   


    band_width=176
    line_width=band_width/expected_dims[0]
    start_wavelength=962


        


    mask_risley=create_risley_pattern(expected_dims,
                              line_width,
                              start_wavelength,
                              tf,
                              PRF,
                              w,
                              w2,
                              w3,
                              w4,
                              a,
                              number_of_prisms,
                              original_volume=None,
                              maximum_transmittance=None,
                              minimum_transmittance=None,
                              sigma=None,
                              plot_mask=True)
    


    plt.imshow(mask_risley[:,:,50],cmap='gray')
    plt.show()
    return mask_risley
########################################################################################
#######################
#######################
#######################                   RISLEY BEAM STEERING
#######################
#######################
########################################################################################
def create_risley_beam_steering_mask(expected_dims):

    #Laser Pulse Rate
    PRF=199900
    #Image Capture Time 0.003
    tf=0.03
    
    #angular speed risley 1 rotations per sec
    w=400
    #angula speed risley 2 rotations per sec
    phi=w/0.09
    
    risley_angle=1*(np.pi/180)
    
    shift_step=17
    band_width=176
    line_width=band_width/expected_dims[0]
    start_wavelength=962
    
    mask_risley=get_risley_3D_mask(expected_dims,
                           PRF,
                           tf,
                           w,
                           phi,
                           risley_angle,
                           shift_step,
                           band_width,
                           line_width,
                           start_wavelength)
    return mask_risley
########################################################################################
#######################
#######################
#######################                   BLUE NOISE MASK 1
#######################
#######################
########################################################################################

def create_blue_noise_mask_1(expected_dims,subsampling_percentage):
    blue_noise=generate_shifting_blue_noise(expected_dims)
    mask=generate_binary_blue_noise_mask(blue_noise,subsampling_percentage)
    return mask
########################################################################################
#######################
#######################
#######################                   RANDOM MASK
#######################
#######################
########################################################################################

def create_random_mask(sub_sampling_percentage,expected_dims):
    random_mask= np.random.choice([1, 0], size=expected_dims, p=[1-sub_sampling_percentage, sub_sampling_percentage])
    total=random_mask.sum()
    missing_data=(100-(total*100)/(random_mask.shape[0]*random_mask.shape[1]*random_mask.shape[2]))
    print(missing_data)
    return random_mask
########################################################################################
#######################
#######################
#######################                   BLUE NOISE MASK
#######################
#######################
########################################################################################
def create_blue_noise_mask(expected_dims,kernel,subsampling_percentage):
    
    blue_noise_cube_normalized=kernel/np.max(kernel)
    blue_noise_cube_normalized_shape=blue_noise_cube_normalized.shape
    
    axis_0=int(np.ceil(expected_dims[0]/blue_noise_cube_normalized_shape[0]))
    axis_1=int(np.ceil(expected_dims[1]/blue_noise_cube_normalized_shape[1]))
    axis_2=int(np.ceil(expected_dims[2]/blue_noise_cube_normalized_shape[2]))
    
    
    
    
    concat1 = np.concatenate(tuple([blue_noise_cube_normalized for i in range(axis_0)]), axis=0)
    concat2 =np.concatenate(tuple([concat1 for i in range(axis_1)]),axis=1)
    concat3 =np. concatenate(tuple([concat2 for i in range(axis_2)]),axis=2)
    
    blue_noise_mask = concat3[0:expected_dims[0],0:expected_dims[1],0:expected_dims[2]]
    
    
    
    binary_blue_noise_mask = blue_noise_mask > subsampling_percentage
    binary_blue_noise_mask = binary_blue_noise_mask*1
    total = binary_blue_noise_mask.sum()
    
    
    missing_data=(100-(total*100)/(blue_noise_mask.shape[0]*blue_noise_mask.shape[1]*blue_noise_mask.shape[2]))
    print(missing_data)
    
    return blue_noise_mask,binary_blue_noise_mask

########################################################################################
#######################
#######################
#######################                   RASTER SCAN MASK
#######################
#######################
########################################################################################
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

    total = volume_mask.sum() 
    missing_data=(100-(total*100)/(volume_mask.shape[0]*volume_mask.shape[1]*volume_mask.shape[2]))
    print(missing_data)

    return volume_mask
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
    w_div_factor = 200
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
                if(sub_volume.shape!=(512,200,16)):
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
    
    

# original_volume =load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1048.pkl')
# subsampled_image = np.multiply(mask,original_volume).astype(np.uint8)
    
    
    


def get_volume_paths():
    amd_eyes_paths = glob("../oct_original_volumes/AMD/*.mat", recursive = True)
    control_eyes_paths= glob("../oct_original_volumes/Control/*.mat", recursive = True)
    all_paths = amd_eyes_paths+control_eyes_paths
    random.shuffle(all_paths)
    return all_paths

########################################################################################
#######################
#######################
#######################                   GENERATE DATASET
#######################
#######################
########################################################################################

def generate_dataset(denoised_dataset_folder_path,
                     generate_ground_truth_denoised,
                     subsampling_method,
                     dataset_folder,
                     training_txt_path,
                     testing_txt_path,
                     mask_path):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        
    if(mask_path):
        mask=load_obj(mask_path)
    else:
        if(subsampling_method=='raster_subsampling'):
            print('raster scan...')
            mask = generate_mask(percentage=sub_sampling_percentage/100,volume_x=1000,volume_y=512,volume_z=100)
        elif(subsampling_method== 'risley_subsampling_4_prisms'):
            print('Risley beam steering 4_prisms scan...')
            mask=create_risley_beam_steering_mask_4_prisms(expected_dims=(512,1000,100),desired_transmittance=(100-sub_sampling_percentage))
        elif(subsampling_method== 'risley_subsampling'):
            print('Risley beam steering scan...')
            mask=create_risley_beam_steering_mask(expected_dims=(512,1000,100))
            
        elif(subsampling_method== 'random_subsampling'):
            print('random scan...')
            mask=create_random_mask(sub_sampling_percentage/100,expected_dims=(512,1000,100))

        elif(subsampling_method=='blue_noise_subsampling'):
            print('blue nosise BIG...')    
            mask=create_blue_noise_mask_1(expected_dims=(512,1000,100),subsampling_percentage=sub_sampling_percentage/100)
        
        save_obj(mask,'./'+dataset_folder+'/mask'+str(sub_sampling_percentage))
    
    
    if(training_txt_path and testing_txt_path):
        with open(training_txt_path) as f:
            lines = f.readlines()
        train_volumes_paths=lines
        with open(testing_txt_path) as f:
            lines = f.readlines()
        test_volumes_paths=lines
    else:
        all_paths=get_volume_paths()
        training_total= int(np.floor(len(all_paths)*0.8))
        
        train_volumes_paths=all_paths[0:training_total]
        test_volumes_paths=all_paths[training_total:]
        with open('./'+dataset_folder+'/train_volumes_paths.txt', 'w') as f:
            f.write('\n'.join(train_volumes_paths))
        with open('./'+dataset_folder+'/test_volumes_paths.txt', 'w') as f:
            f.write('\n'.join(test_volumes_paths))
            
            
    
    
    
    volume_number=0
    subsampled_volumes_dataset_train = h5py.File('./'+dataset_folder+'/training_subsampled_volumes.h5', 'w')
    volumes_dataset_train = h5py.File('./'+dataset_folder+'/training_ground_truth.h5', 'w')
        
    for volume_path in train_volumes_paths:
        volume_path=volume_path.strip('\n')
        print(volume_path)
        try: 
            volume = read_data(volume_path)
        
            subsampled_image = np.multiply(mask,volume).astype(np.uint8)

            
            name='original_train_vol_'+str(volume_number)
            if(generate_ground_truth_denoised):
                denoised_volume=find_denoised_volume(volume_path,denoised_dataset_folder_path)
                #denoised_volume=median_filter_3D(volume,40,5)
                #plt.imshow(denoised_volume[:,:,11],cmap='gray')
                #plt.show()
                #plt.imshow(volume[:,:,11],cmap='gray')
                #plt.show()
                extract_sub_volumes(denoised_volume,name,volumes_dataset_train)
            else:
                extract_sub_volumes(volume,name,volumes_dataset_train)
                
            name='subsampled_train_vol_'+str(volume_number)
            extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_train)
            
            volume_number+=1
        
        except:
            print('WRONG dimentions'+volume_path)

    
    
    subsampled_volumes_dataset_train.close()  
    volumes_dataset_train.close()
    
    
    
    
    volume_number=0
    subsampled_volumes_dataset_test = h5py.File('./'+dataset_folder+'/testing_random25_subsampled_volumes.h5', 'w')
    volumes_dataset_test = h5py.File('./'+dataset_folder+'/testing_random25_ground_truth.h5', 'w')
        
        
    for volume_path in test_volumes_paths:
            
            try:
                volume_path=volume_path.strip('\n')
                print(volume_path)
        
                volume = read_data(volume_path)
                
                subsampled_image = np.multiply(mask,volume).astype(np.uint8)
                
                
                name='original_test_vol_'+str(volume_number)
                if(generate_ground_truth_denoised):
                    denoised_volume=find_denoised_volume(volume_path,denoised_dataset_folder_path)
                    #denoised_volume=median_filter_3D(volume,40,5)
                    #plt.imshow(denoised_volume[:,:,11],cmap='gray')
                    #plt.show()
                    #plt.imshow(volume[:,:,11],cmap='gray')
                    #plt.show()
                    extract_sub_volumes(denoised_volume,name,volumes_dataset_test)
                else:
                    extract_sub_volumes(volume,name,volumes_dataset_test)
                
                name='subsampled_test_vol_'+str(volume_number)
                extract_sub_volumes(subsampled_image,name,subsampled_volumes_dataset_test)
                
                volume_number+=1
            
            except:
                print('WRONG dimentions'+volume_path)

            
            
    subsampled_volumes_dataset_test.close()  
    volumes_dataset_test.close()

dataset_folder='TEST_DATASET'
denoised_dataset_folder_path='./DATASET_DENOISED'
generate_ground_truth_denoised=False
# training_txt_path='./TEST_DATASET_FIXED_MASK/train_volumes_paths.txt'
# testing_txt_path='./TEST_DATASET_FIXED_MASK/test_volumes_paths.txt'
# mask_path='./TEST_DATASET_FIXED_MASK/mask75.pkl'
generate_dataset(denoised_dataset_folder_path,
                 generate_ground_truth_denoised,
                 subsampling_method,
                 dataset_folder,
                 training_txt_path='',
                 testing_txt_path='',
                 mask_path='')   

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
# import h5py
# import matplotlib.pyplot as plt
# p1='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/RISLEY_BEAM_STEERING_25_TRANSMITTANCE_DATASET/training_ground_truth.h5'
# p2='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/RISLEY_BEAM_STEERING_25_TRANSMITTANCE_DATASET/training_subsampled_volumes.h5'
# f = h5py.File(p1, 'r')
# name=list(f.keys())[300]
# value=np.array(f.get(name))
# f.close()

# plt.imshow(value[:,:,0],cmap='gray')
# plt.show()