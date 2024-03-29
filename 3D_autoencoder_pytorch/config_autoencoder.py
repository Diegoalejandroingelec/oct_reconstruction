#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:40:28 2022

@author: diego
"""
import torch
# Number of training epochs
num_epochs = 5

# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.999)


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 16

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

sub_volumes_dim=(512,64,16)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Dynamically adjust the learning rate policy
lr_scheduler_step_size = num_epochs // 2
lr_scheduler_gamma = 0.1

results_dir='MOTION_MAXIMUM_10X_10Y_CROP_DATASET_GT_MEDIAN_FILTER'
resume_model_path=''
################TRAINING#################
subsampled_volumes_path='../MOTION_MAXIMUM_10X_10Y_CROP_DATASET/training_subsampled_volumes.h5'
original_volumes_path='../MOTION_MAXIMUM_10X_10Y_CROP_DATASET/training_ground_truth.h5'

################TESTING#################
subsampled_volumes_path_test='../MOTION_MAXIMUM_10X_10Y_CROP_DATASET/testing_subsampled_volumes.h5'
original_volumes_path_test='../MOTION_MAXIMUM_10X_10Y_CROP_DATASET/testing_ground_truth.h5'
