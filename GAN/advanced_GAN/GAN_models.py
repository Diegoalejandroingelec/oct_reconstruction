#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:10:14 2022

@author: diego
"""


import torch
from torch import nn
import config

__all__ = [
    "ResidualConvBlock3D",
    "Discriminator", "Generator",
]


class ResidualConvBlock3D(nn.Module):
    """Implements residual conv function.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock3D, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True),
            nn.BatchNorm3d(channels),
            nn.PReLU(),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)

        return out





class Discriminator(nn.Module):
    def __init__(self,ngpu) -> None:
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # INPUT SIZE 1 x 512 x 64 x 16
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 64 x 256 x 32 x 8
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 128 x 128 x 16 x 4
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 256 x 64 x 8 x 2
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
            # STATE SIZE 512 x 32 x 4 x 1
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * config.sub_volumes_dim[1]//16* config.sub_volumes_dim[2]//16 * config.sub_volumes_dim[3]//16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self,ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # First conv layer.
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(9, 9, 9), padding='same'),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(config.n_residual_blocks):
            trunk.append(ResidualConvBlock3D(64))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
        )


        # Output layer.
        self.conv_block3 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(9, 9, 9), padding='same')

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.conv_block3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
