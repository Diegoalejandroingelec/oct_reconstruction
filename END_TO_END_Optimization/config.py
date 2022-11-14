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
batch_size = 2

# Batch size for testing
batch_size_testing = 2

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

sub_volumes_dim=(512,200,16)
original_dims=(512,1000,100)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Dynamically adjust the learning rate policy
lr_scheduler_step_size = num_epochs // 2
lr_scheduler_gamma = 0.1

results_dir='END_TO_END_OPTIMIZATION_big'
resume_model_path='./END_TO_END_OPTIMIZATION/BEST_MODEL_autoencoder_0.pth.tar'

############RESUME MODEL SPEEDS##########
resume_model_speeds_path='./END_TO_END_OPTIMIZATION/BEST_MODEL_speeds_epoch_0.pth.tar'
################TRAINING#################
subsampled_volumes_path='../TEST_DATASET/training_subsampled_volumes.h5'
original_volumes_path='../TEST_DATASET/training_ground_truth.h5'


################TESTING#################
subsampled_volumes_path_test='../TEST_DATASET/testing_subsampled_volumes.h5'
original_volumes_path_test='../TEST_DATASET/testing_ground_truth.h5'
