#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:11:22 2022

@author: diego
"""
import torch

# Incremental training and migration training
resume_d = ""
resume_g = ""

# Number of residual blocks in Generator #16
n_residual_blocks=16
# Experiment name, easy to save weights and log files
exp_name = "GAN_OCT"

# How many iterations to print the training result
print_frequency = 200
epochs=5

ngpu=1

sub_volumes_dim=(1,512,64,16)
#sub_volumes_dim=(1, 64,64,16)
subsampled_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/random_sub_sampling_Data/training_random_subsampled_volumes.h5'
original_volumes_path='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/random_sub_sampling_Data/training_random_ground_truth.h5'

subsampled_volumes_path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/random_sub_sampling_Data/testing_random_subsampled_volumes.h5'
original_volumes_path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/random_sub_sampling_Data/testing_random_ground_truth.h5'

batch_size=4
num_workers=4

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.999)

# Dynamically adjust the learning rate policy
lr_scheduler_step_size = epochs // 2
lr_scheduler_gamma = 0.1

# Loss function weight

content_weight = 1.0
adversarial_weight = 0.001