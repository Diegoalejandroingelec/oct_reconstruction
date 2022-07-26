#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:28:46 2022

@author: diego
"""
import h5py
import numpy as np
import torch
from torch.utils import data

class H5LoadDataset(data.Dataset):

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

        # get original
        y = self.get_ground_truth(name)

        if self.transform:
            y = self.transform(y)

        y = torch.from_numpy(y)

        x=torch.unsqueeze(x,0)
        y=torch.unsqueeze(y,0)
        return {"subsampled": x, "original": y}
    
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
    return volume.astype(np.float32)/255


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)