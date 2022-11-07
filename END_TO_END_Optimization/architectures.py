#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:05:27 2022

@author: diego
"""

import torch
import torch.nn as nn
__all__ = [
    "Autoencoder",
    "Risley_Speeds"
]

class Autoencoder(nn.Module):
    def __init__(self,ngpu):
        super(Autoencoder,self).__init__()

        layers = [32,32,32,32]
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
            nn.Conv3d(1,layers[0],kernel_size=3,padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm3d(layers[0])
        )

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(layers[s],layers[s+1],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[s+1])
            )  for s in range(len(layers) - 1)
        )

        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose3d(layers[len(layers)-1-s],layers[len(layers)-2-s],kernel_size=3,padding=[0,0,0]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm3d(layers[len(layers)-2-s]),
            )  for s in range(len(layers) - 1)
        )

        self.output = nn.Sequential(
            nn.Conv3d(layers[0],1,kernel_size=3,padding='same'),
            nn.Tanh()
        )


    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = self.input(x)

        for i,j in enumerate(self.encoder):
            x = self.encoder[i](x)

        for i,j in enumerate(self.decoder):
            x = self.decoder[i](x)
        
        x = self.output(x)

        return x
    
    
class Risley_Speeds(nn.Module):
    def __init__(self,ngpu) -> None:
        super(Risley_Speeds, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # INPUT SIZE 1 x 512 x 64 x 16
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 64 x 256 x 32 x 8
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 128 x 128 x 16 x 4
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 256 x 64 x 8 x 2
            
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
            
            # STATE SIZE 512 x 32 x 4 x 1
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(212992, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x,1)
        out = self.features(x)
        out = self.classifier(out)

        return out
        
        