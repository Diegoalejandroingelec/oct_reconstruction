#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:53:40 2022

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
from Autoencoder_Architecture import Autoencoder
import config_autoencoder
from torch.optim import lr_scheduler

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

    def __init__(self, file_path, ground_truth_path,prefix_for_test, transform=None):
        super().__init__()
        self.file_path = file_path
        self.ground_truth_path = ground_truth_path
        self.transform = transform
        self.prefix_for_test=prefix_for_test
        
    def __getitem__(self, index):
        # get data
        x,name = self.get_data(index)
        if self.transform:
            x = self.transform(x)

        x = torch.from_numpy(x)

        # get label
        y = self.get_ground_truth(name)
        
        if self.transform:
            y = self.transform(y)
        
        y = torch.from_numpy(y)
        return (x, y)
    
    def get_ground_truth(self,reference_name):
        f_gt = h5py.File(self.ground_truth_path, 'r')
        name = self.prefix_for_test+'_'+ '_'.join(reference_name.split('_')[-3:])
        value=np.array(f_gt.get(name))
        f_gt.close()
        return value
    
    def __len__(self):
        return self.get_info()
    
    def get_data(self,index):
        f = h5py.File(self.file_path, 'r')
        name=list(f.keys())[index]
        value=np.array(f.get(name))
        f.close()
        return value,name
    
    def get_info(self):
        f = h5py.File(self.file_path, 'r')
        info=len(list(f.keys()))
        f.close()
        return info
    
def normalize(volume):
    return (volume.astype(np.float32)-(np.max(volume.astype(np.float32))/2))/(np.max(volume.astype(np.float32))/2)


h5_dataset=HDF5Dataset(config_autoencoder.subsampled_volumes_path,
                       config_autoencoder.original_volumes_path,
                       'original_train',
                       normalize)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(h5_dataset,
                                         batch_size=config_autoencoder.batch_size,
                                         shuffle=True,
                                         num_workers=config_autoencoder.workers)

# train_batch = next(iter(dataloader))

# plt.imshow(np.squeeze(np.array(train_batch[1][0,:,:,0].cpu())), cmap="gray")
# plt.imshow(np.squeeze(np.array(train_batch[0][0,:,:,0].cpu())), cmap="gray")


h5_dataset_test=HDF5Dataset(config_autoencoder.subsampled_volumes_path_test,
                            config_autoencoder.original_volumes_path_test,
                            'original_test',
                            normalize)
# Create the dataloader
dataloader_test = torch.utils.data.DataLoader(h5_dataset_test,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config_autoencoder.workers)

# test_batch = next(iter(dataloader_test))

# plt.imshow(np.squeeze(np.array(test_batch[1][0,:,:,0].cpu())), cmap="gray")
# plt.imshow(np.squeeze(np.array(test_batch[0][0,:,:,0].cpu())), cmap="gray")




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
        
def define_optimizer(netG: nn.Module) -> [optim.Adam]:
    optimizer = optim.Adam(netG.parameters(), lr=config_autoencoder.model_lr, betas=config_autoencoder.model_betas)
    return optimizer   
     
def define_scheduler(optimizer: optim.Adam) -> [lr_scheduler.StepLR]:
    scheduler = lr_scheduler.StepLR(optimizer, config_autoencoder.lr_scheduler_step_size, config_autoencoder.lr_scheduler_gamma)
    return scheduler 
     



netG = Autoencoder(config_autoencoder.ngpu).to(config_autoencoder.device)
# Handle multi-gpu if desired
if (config_autoencoder.device.type == 'cuda') and (config_autoencoder.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(config_autoencoder.ngpu)))





criterion = nn.L1Loss()
criterion_for_testing=nn.L1Loss()
print("Define all loss functions successfully.")
optimizer=define_optimizer(netG)
print("Define all optimizer functions successfully.")
scheduler = define_scheduler(optimizer)
print("Define all optimizer scheduler functions successfully.")



if(config_autoencoder.resume_model_path):
    # Load checkpoint model
    checkpoint = torch.load(config_autoencoder.resume_model_path, map_location=lambda storage, loc: storage)
    # Restore the parameters in the training node to this point
    start_epoch = checkpoint["epoch"]
    best_psnr = checkpoint["best_psnr"]
    best_ssim = checkpoint["best_ssim"]
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = netG.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    netG.load_state_dict(model_state_dict)
    # Load the optimizer model
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Load the scheduler model
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Loaded `{config_autoencoder.resume_model_path}` resume netG model weights successfully. "
          f"Resume training from epoch {start_epoch}.")
    summary(netG, config_autoencoder.sub_volumes_dim)
else:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    print('Training from scratch...')
    netG.apply(weights_init)
    summary(netG, config_autoencoder.sub_volumes_dim)
    start_epoch=0




if not os.path.exists(config_autoencoder.results_dir):
    os.makedirs(config_autoencoder.results_dir)


print("Starting Training Loop...")
# For each epoch
losses = []
losses_val=[]
best_psnr=0
best_ssim=0
for epoch in range(start_epoch, config_autoencoder.num_epochs):
    # For each batch in the dataloader
    for i, data_train in enumerate(dataloader, 0):  
        inputs = data_train[0].to(config_autoencoder.device, dtype=torch.float)
        targets = data_train[1].to(config_autoencoder.device, dtype=torch.float)
        
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        reconstructions = netG(inputs)
        # calculate loss
        loss = criterion(reconstructions, torch.unsqueeze(targets,1))
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, config_autoencoder.num_epochs, i, len(dataloader),loss.item()))
            
        losses.append(loss.item())

    # Update LR
    scheduler.step()
    
    test_losses=[]
    psnr_list=[]
    ssim_list=[]
    print('Evaluation...')
    
    for j, data_test in enumerate(dataloader_test, 0):  
         inputs_test = data_test[0].to(config_autoencoder.device, dtype=torch.float)
         targets_test = data_test[1].to(config_autoencoder.device, dtype=torch.float)
         # compute the model output
         reconstructions_test = netG(inputs_test)
         # calculate loss
         
         loss_test = criterion_for_testing(reconstructions_test, torch.unsqueeze(targets_test,1))
         test_losses.append(loss_test.item())
         
         reconstructed_8bit=np.squeeze(((reconstructions_test.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
         original_8bit=np.squeeze(((targets_test.cpu().detach().numpy()*127.5)+127.5).astype(np.uint8))
         # Statistical loss value for terminal data output
         psnr_value = compute_PSNR(reconstructed_8bit, original_8bit)
         ssim_value = ssim(reconstructed_8bit, original_8bit)
         psnr_list.append(psnr_value)
         ssim_list.append(ssim_value)
         
         
         if j % 5000 == 0:
             print(j)

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
                os.path.join(config_autoencoder.results_dir, f"model_epoch_{epoch}.pth.tar"))
    
    
    if is_best:
        best_psnr=current_psnr
        best_ssim=current_ssim
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": netG.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                    os.path.join(config_autoencoder.results_dir, f"BEST_MODEL_{epoch}.pth.tar"))
        
        
    losses_val.append(current_loss)
    
    
save_obj(losses,os.path.join(config_autoencoder.results_dir, 'train_losses' ))
save_obj(losses_val,os.path.join(config_autoencoder.results_dir, 'test_losses' ))      
      


# def load_obj(name):
#     with open( name, 'rb') as f:
#         return pickle.load(f)
    
# L=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/3D_autoencoder_pytorch/train_losses_blue_noise.pkl')    
# plt.figure(figsize=(10,5))
# plt.title("Autoencoder Loss During Training")
# plt.plot(L)

# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

