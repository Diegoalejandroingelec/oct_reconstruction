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
batch_size = 4

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

sub_volumes_dim=(512,64,16)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Dynamically adjust the learning rate policy
lr_scheduler_step_size = num_epochs // 2
lr_scheduler_gamma = 0.1

results_dir='TRAINING_RANDOM_75'
resume_model_path=''
################TRAINING#################
subsampled_volumes_path='../random_sub_sampling_Data/training_random_subsampled_volumes.h5'
original_volumes_path='../random_sub_sampling_Data/training_random_ground_truth.h5'

################TESTING#################
subsampled_volumes_path_test='../random_sub_sampling_Data/testing_random_subsampled_volumes.h5'
original_volumes_path_test='../random_sub_sampling_Data/testing_random_ground_truth.h5'