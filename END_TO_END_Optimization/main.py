#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:53:01 2022

@author: diego
"""

import os
import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from architectures import Autoencoder, Risley_Speeds
import config
from torch.optim import lr_scheduler
from risley_varying_all_parameters import create_risley_pattern 



train_with_motion=False

def create_3D_mask(w1,w2,w3,w4,original_volume=None,train_with_motion=False):
    
    expected_dims=config.sub_volumes_dim
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
                              x_translation=100,
                              y_translation=8,
                              x_factor_addition=0.3,
                              y_factor_addition=0.15,
                              tf=8.192,
                              PRF=5000000,#3000000,
                              a=10*(np.pi/180),
                              number_of_prisms=4,
                              maximum_transmittance=0.43,
                              minimum_transmittance=0.0,
                              sigma=150,
                              transmittance_distribution_fn='ga',
                              number_of_laser_sweeps=10,
                              steps_before_centering=10,
                              hand_tremor_period=1/9,
                              laser_time_between_sweeps=7.314285714285714e-05,
                              x_factor=50,
                              y_factor=50,
                              generate_volume_with_motion=train_with_motion,
                              apply_motion=train_with_motion,
                              plot_mask=False)
    if(train_with_motion):
        return mask_risley[0],mask_risley[1]
    else:
        return mask_risley[0],mask_risley[1]


def compute_PSNR(original,reconstruction,bit_representation=8):
    mse = nn.MSELoss()
    x=torch.from_numpy(original.astype('float'))
    y=torch.from_numpy(reconstruction.astype('float'))    
    MSE=mse(x, y)
    MSE=MSE.item()
    MAXI=np.power(2,bit_representation)-1
    PSNR=20*np.log10(MAXI)-10*np.log10(MSE)
    return PSNR

def save_obj(obj,path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class HDF5Dataset(data.Dataset):

    def __init__(self, ground_truth_path):
        super().__init__()
        self.ground_truth_path = ground_truth_path
        
    def __getitem__(self, index):
        # get data
        return self.get_data(index)
    
    
    def __len__(self):
        return self.get_info()
    
    def get_data(self,index):
        f = h5py.File(self.ground_truth_path, 'r')
        name=list(f.keys())[index]
        value=np.array(f.get(name))
        f.close()
        return torch.from_numpy(value)
    
    def get_info(self):
        f = h5py.File(self.ground_truth_path, 'r')
        info=len(list(f.keys()))
        f.close()
        return info
    
def normalize(volume):
    if((np.max(volume.astype(np.float32))/2)==0):
        return np.ones(config.sub_volumes_dim)*-1
    
    return (volume.astype(np.float32)-(np.max(volume.astype(np.float32))/2))/(np.max(volume.astype(np.float32))/2)


h5_dataset=HDF5Dataset(config.original_volumes_path)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(h5_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=config.workers)

#train_batch = next(iter(dataloader))

# plt.imshow(np.squeeze(np.array(train_batch[1][4,:,:,0].cpu())), cmap="gray")
# plt.show()
# plt.imshow(np.squeeze(np.array(train_batch[0][4,:,:,0].cpu())), cmap="gray")
# plt.show()

h5_dataset_test=HDF5Dataset(config.original_volumes_path_test)
# Create the dataloader
dataloader_test = torch.utils.data.DataLoader(h5_dataset_test,
                                              batch_size=config.batch_size_testing,
                                              shuffle=True,
                                              num_workers=config.workers)

# test_batch = next(iter(dataloader_test))

# plt.imshow(np.squeeze(np.array(test_batch[1][0,:,:,0].cpu())), cmap="gray")
# plt.show()
# plt.imshow(np.squeeze(np.array(test_batch[0][0,:,:,0].cpu())), cmap="gray")
# plt.show()



# custom weights initialization called on netG 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def define_optimizer(netG: nn.Module, speeds_generator: nn.Module) -> [optim.Adam,optim.Adam]:
    optimizer = optim.Adam(netG.parameters(), lr=config.model_lr, betas=config.model_betas)
    optimizer_speeds = optim.Adam(speeds_generator.parameters(), lr=config.model_lr, betas=config.model_betas)
    return optimizer,optimizer_speeds  
     
def define_scheduler(optimizer: optim.Adam) -> [lr_scheduler.StepLR]:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)
    return scheduler 

     


speeds_generator=Risley_Speeds(config.ngpu).to(config.device)
if (config.device.type == 'cuda') and (config.ngpu > 1):
    speeds_generator = nn.DataParallel(speeds_generator, list(range(config.ngpu)))
# summary(speeds_generator, config.sub_volumes_dim)


netG = Autoencoder(config.ngpu).to(config.device)
# Handle multi-gpu if desired
if (config.device.type == 'cuda') and (config.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(config.ngpu)))




speed_criterion= nn.L1Loss()
criterion = nn.L1Loss()



criterion_for_testing=nn.L1Loss()
print("Define all loss functions successfully.")
optimizer,optimizer_speeds=define_optimizer(netG,speeds_generator)
print("Define all optimizer functions successfully.")
scheduler = define_scheduler(optimizer)
print("Define all optimizer scheduler functions successfully.")



if(config.resume_model_path):
    # Load checkpoint model
    checkpoint = torch.load(config.resume_model_path, map_location=lambda storage, loc: storage)
    # Restore the parameters in the training node to this point
    start_epoch = checkpoint["epoch"]
    best_psnr = checkpoint["best_psnr"]
    best_ssim = checkpoint["best_ssim"]
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = netG.state_dict()
    # new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items() if
    #                   k.replace('module.', '') in model_state_dict.keys() and v.size() == model_state_dict[k.replace('module.', '')].size()}
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    netG.load_state_dict(model_state_dict)
    # Load the optimizer model
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Load the scheduler model
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Loaded `{config.resume_model_path}` resume netG model weights successfully. "
          f"Resume training from epoch {start_epoch}.")
    summary(netG, config.sub_volumes_dim)
else:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    print('AUTOENCODER Training from scratch...')
    netG.apply(weights_init)
    summary(netG, config.sub_volumes_dim)
    start_epoch=0
    best_psnr=0
    best_ssim=0


if(config.resume_model_speeds_path):
    # Load checkpoint model
    checkpoint = torch.load(config.resume_model_speeds_path, map_location=lambda storage, loc: storage)
    # Restore the parameters in the training node to this point
   
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = speeds_generator.state_dict()
    # new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items() if
    #                   k.replace('module.', '') in model_state_dict.keys() and v.size() == model_state_dict[k.replace('module.', '')].size()}
    
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    speeds_generator.load_state_dict(model_state_dict)
    # Load the optimizer model
    optimizer_speeds.load_state_dict(checkpoint["optimizer"])
    
    print(f"Loaded `{config.resume_model_speeds_path}` resume speeds_generator model weights successfully.")
    summary(speeds_generator, config.sub_volumes_dim)
else:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    print('SPEEDS GENERATOR Training from scratch...')
    speeds_generator.apply(weights_init)
    summary(speeds_generator, config.sub_volumes_dim)




if not os.path.exists(config.results_dir):
    os.makedirs(config.results_dir)


print("Starting Training Loop...")
# For each epoch
losses = []
losses_val=[]


train_speeds=True

for epoch in range(start_epoch, config.num_epochs):
    # For each batch in the dataloader
    for i, data_train in enumerate(dataloader, 0):  
        
        #inputs = data_train[0].to(config.device, dtype=torch.float)
        targets = data_train.to(config.device, dtype=torch.float)
        
        
        if(train_speeds):
            #print('training speeds generator')
            # During speeds generator training, turn off autoencoder backpropagation
            for autoencoder_parameters in netG.parameters():
                autoencoder_parameters.requires_grad = False
            # During speeds generator training, turn on speeds_generator backpropagation
            for speeds_generator_parameters in speeds_generator.parameters():
                speeds_generator_parameters.requires_grad = True
            # Initialize the discriminator model gradients
            speeds_generator.zero_grad(set_to_none=True)
            
            
        else:
            #print('training autoencoder')
            # During autoencoder training, turn on autoencoder backpropagation
            for autoencoder_parameters in netG.parameters():
                autoencoder_parameters.requires_grad = True
            # During autoencoder training, turn off speeds generator backpropagation
            for speeds_generator_parameters in speeds_generator.parameters():
                speeds_generator_parameters.requires_grad = False
            # Initialize the discriminator model gradients
            netG.zero_grad(set_to_none=True) 
           
            
        # NORMALIZE IMAGE BEFORE PREDICTING SPEEDS
        gt_normalized=torch.tensor(np.array([normalize(gt.cpu().detach().numpy()) for gt in targets]),requires_grad=True).to(config.device,dtype=torch.float)
        speeds=speeds_generator(gt_normalized).cpu().detach().numpy()
        speeds_avg=np.mean(speeds,0)
        
        
        
        
        if(train_with_motion):
            subsampled_volumes_normalized=[]
            for cube in targets:
                mask,subsampled_volume=create_3D_mask(w1=speeds_avg[0]*100000,
                                w2=speeds_avg[1]*100000,
                                w3=speeds_avg[2]*100000,
                                w4=speeds_avg[3]*100000,
                                original_volume=cube.cpu().detach().numpy(),
                                train_with_motion=True)
                subsampled_volumes_normalized.append(normalize(subsampled_volume))
            subsampled_volumes_normalized=np.array(subsampled_volumes_normalized)
                 
        else:
            mask,total_transmittance=create_3D_mask(w1=speeds_avg[0]*100000,
                            w2=speeds_avg[1]*100000,
                            w3=speeds_avg[2]*100000,
                            w4=speeds_avg[3]*100000,
                            original_volume=None)
            
            subsampled_volumes=[np.multiply(mask,cube.cpu().detach().numpy()) for cube in targets]
            subsampled_volumes_normalized=np.array([normalize(subsampled_volume) for subsampled_volume in subsampled_volumes])
            
        # mask=create_3D_mask(w1=speeds_avg[0]*100000,
        #                 w2=speeds_avg[1]*100000,
        #                 w3=speeds_avg[2]*100000,
        #                 w4=speeds_avg[3]*100000,
        #                 original_volume=None)
        
        #print(mask.shape)
        # clear the gradients
        if(train_speeds):
            optimizer_speeds.zero_grad()
        else:
            optimizer.zero_grad()
        
                
        reconstructions = netG(torch.tensor(subsampled_volumes_normalized,
                                                            requires_grad=True).to(config.device,
                                                                                  dtype=torch.float))
                                                                                   
                                                                                   
                                                                                   
        ground_truths=torch.squeeze(targets).cpu().detach().numpy()
        ground_truth_normalized=torch.tensor(np.array([normalize(ground_truth) for ground_truth in ground_truths]),
                                                            requires_grad=True).to(config.device,
                                                                                  dtype=torch.float)
                                                          
        
        # update model weights
        
        if(train_speeds):
            loss_speeds=speed_criterion(reconstructions,torch.unsqueeze(ground_truth_normalized,1))
            loss_speeds.backward()
            optimizer_speeds.step()
            
            train_speeds=False
            
            # Output training stats
            if i % 5 == 0:
                psnr_value_train_=[]
                ssim_value_train_=[]
                for r,o in zip(reconstructions,ground_truth_normalized):
                    reconstructed_8bit=np.squeeze(((r.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
                    original_8bit=np.squeeze(((o.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
                    # Statistical loss value for terminal data output
                    psnr_value_train_.append(compute_PSNR(reconstructed_8bit, original_8bit))
                    ssim_value_train_.append(ssim(reconstructed_8bit, original_8bit))
                    
                    
                print('[%d/%d][%d/%d]\tLoss_speeds: %.4f  PSNR: %.4f SSIM:%.4f Transmittance: %.4f' % (epoch,
                                                                                   config.num_epochs,
                                                                                   i,
                                                                                   len(dataloader),
                                                                                   loss_speeds.item(),
                                                                                   np.array(psnr_value_train_).mean(),
                                                                                   np.array(ssim_value_train_).mean(),
                                                                                   total_transmittance))
                
            
            
        else:
            
            loss = criterion(reconstructions, torch.unsqueeze(ground_truth_normalized,1))
            loss.backward()
            optimizer.step()
            
            train_speeds=True
            
            
        
            # Output training stats
            if i % 5 == 0:
                psnr_value_train_=[]
                ssim_value_train_=[]
                for r,o in zip(reconstructions,ground_truth_normalized):
                    reconstructed_8bit=np.squeeze(((r.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
                    original_8bit=np.squeeze(((o.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
                    # Statistical loss value for terminal data output
                    psnr_value_train_.append(compute_PSNR(reconstructed_8bit, original_8bit))
                    ssim_value_train_.append(ssim(reconstructed_8bit, original_8bit))
                    
                    
                print('[%d/%d][%d/%d]\tLoss_autoencoder: %.4f  PSNR: %.4f SSIM:%.4f Transmittance: %.4f' % (epoch,
                                                                                   config.num_epochs,
                                                                                   i,
                                                                                   len(dataloader),
                                                                                   loss_speeds.item(),
                                                                                   np.array(psnr_value_train_).mean(),
                                                                                   np.array(ssim_value_train_).mean(),
                                                                                   total_transmittance))

                
            losses.append(loss.item())
    # Update LR
    scheduler.step()
    
    test_losses=[]
    psnr_list=[]
    ssim_list=[]
    print('Saving weights...')
    torch.save({"epoch": epoch + 1,
                "best_psnr": np.mean(psnr_value_train_),
                "best_ssim": np.mean(ssim_value_train_),
                "state_dict": netG.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
                os.path.join(config.results_dir, f"autoencoder_model_epoch_{epoch}.pth.tar"))


    torch.save({"state_dict": speeds_generator.state_dict(),
                "optimizer": optimizer_speeds.state_dict()},
                os.path.join(config.results_dir, f"speeds_model_epoch_{epoch}.pth.tar"))
    
    print('Evaluation...')
    
    for j, data_test in enumerate(dataloader_test, 0):  
         targets_test = data_test.to(config.device, dtype=torch.float)
         
         speeds_pred=speeds_generator(targets_test).cpu().detach().numpy()
         speeds_avg_test=np.mean(speeds_pred,0)
         
         if (train_with_motion):
             mask_test,subsampled_volume_test=create_3D_mask(w1=speeds_avg_test[0]*100000,
                             w2=speeds_avg_test[1]*100000,
                             w3=speeds_avg_test[2]*100000,
                             w4=speeds_avg_test[3]*100000,
                             original_volume=np.squeeze(targets_test.cpu().detach().numpy()),
                             train_with_motion=True)
             subsampled_volumes_test_normalized=normalize(subsampled_volume_test)
         else:
             mask_test,total_transmittance=create_3D_mask(w1=speeds_avg_test[0]*100000,
                             w2=speeds_avg_test[1]*100000,
                             w3=speeds_avg_test[2]*100000,
                             w4=speeds_avg_test[3]*100000,
                             original_volume=None)
         
         
             subsampled_volumes_test=[np.multiply(mask_test,cube.cpu().detach().numpy()) for cube in targets_test]
             subsampled_volumes_test_normalized=np.array([normalize(subsampled_volume) for subsampled_volume in subsampled_volumes_test])
             
         
         # compute the model output

         
         reconstructions_test = netG(torch.tensor(subsampled_volumes_test_normalized).to(config.device,
                                                                                   dtype=torch.float))
                                                                                    
         ground_truths_test=torch.squeeze(targets_test).cpu().detach().numpy()
         ground_truth_normalized_test=torch.tensor(np.array([normalize(ground_truth) for ground_truth in ground_truths_test])).to(config.device,
                                                                                  dtype=torch.float)
         # calculate loss
         
         loss_test = criterion_for_testing(torch.squeeze(reconstructions_test), ground_truth_normalized_test)
         test_losses.append(loss_test.item())
         
         for r_t, o_t in zip(reconstructions_test,ground_truth_normalized_test):
             reconstructed_8bit=np.squeeze(((r_t.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
             original_8bit=np.squeeze(((o_t.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
             # Statistical loss value for terminal data output
             psnr_value = compute_PSNR(reconstructed_8bit, original_8bit)
             ssim_value = ssim(reconstructed_8bit, original_8bit)
             psnr_list.append(psnr_value)
             ssim_list.append(ssim_value)
         
         
         if j % 100 == 0:
             total_samples=len(dataloader_test)
             print(f"{j}\{total_samples}  Loss_test={loss_test.item()} PSNR_test={np.mean(psnr_list)} SSIM_test={np.mean(ssim_list)} Transmittance={total_transmittance}")
         if j % 2000 ==0 and j!=0:
             break
         
    current_loss=np.mean(test_losses)
    current_psnr=np.mean(psnr_list)
    current_ssim=np.mean(ssim_list)
    print('VALIDATION LOSS: ',current_loss)
    print('VALIDATION SSIM: ',current_ssim)
    print('VALIDATION PSNR: ',current_psnr)
    
    is_best=current_psnr>best_psnr and current_ssim>best_ssim
    
    
    torch.save({"epoch": epoch + 1,
                "best_psnr": current_psnr,
                "best_ssim": current_ssim,
                "state_dict": netG.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
                os.path.join(config.results_dir, f"autoencoder_model_epoch_{epoch}.pth.tar"))


    torch.save({"state_dict": speeds_generator.state_dict(),
                "optimizer": optimizer_speeds.state_dict()},
                os.path.join(config.results_dir, f"speeds_model_epoch_{epoch}.pth.tar"))
    
    
    if is_best:
        print(speeds_avg*100000)
        save_obj(speeds_avg*100000,os.path.join(config.results_dir, 'BEST_SPEEDS' ))
        best_psnr=current_psnr
        best_ssim=current_ssim
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": netG.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                    os.path.join(config.results_dir, f"BEST_MODEL_autoencoder_{epoch}.pth.tar"))
        
        torch.save({"state_dict": speeds_generator.state_dict(),
                    "optimizer": optimizer_speeds.state_dict()},
                    os.path.join(config.results_dir, f"BEST_MODEL_speeds_epoch_{epoch}.pth.tar"))
        
        
    losses_val.append(current_loss)
    
    
save_obj(losses,os.path.join(config.results_dir, 'train_losses' ))
save_obj(losses_val,os.path.join(config.results_dir, 'test_losses' ))