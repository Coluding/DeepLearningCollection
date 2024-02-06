import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import math
import time
import gc
from typing import Union
from abc import ABC, abstractmethod

from src.computer_vision.segmentation.aspp import ASPP


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) layer as described in DeepLab architectures.
    This layer uses multiple dilated convolutions to capture multi-scale information.

    :param in_ch: Number of input channels.

    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        # Convolution with dilation rate of 1 (normal convolution)
        self.aconv1 = nn.Conv2d(in_ch, 256, 3, dilation=1, padding="same")

        # Convolution with dilation rate of 6
        self.aconv2 = nn.Conv2d(in_ch, 256, 3, dilation=6, padding="same")

        # Convolution with dilation rate of 12
        self.aconv3 = nn.Conv2d(in_ch, 256, 3, dilation=12, padding="same")

        # Convolution with dilation rate of 18
        self.aconv4 = nn.Conv2d(in_ch, 256, 3, dilation=18, padding="same")

        # Convolution with dilation rate of 24
        self.aconv5 = nn.Conv2d(in_ch, 256, 3, dilation=24, padding="same")

        # Batch normalization for concatenated feature maps
        self.bn = nn.BatchNorm2d(256 * 5)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Prediction convolution
        self.pred_conv = nn.Conv2d(256 * 5, out_ch, 1, padding="same")

    def forward(self, x):
        """
        Forward pass through the ASPP layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after ASPP.
        """

        # Pass the input through each dilated convolution and apply ReLU
        out1 = self.relu(self.aconv1(x))
        out2 = self.relu(self.aconv2(x))
        out3 = self.relu(self.aconv3(x))
        out4 = self.relu(self.aconv4(x))
        out5 = self.relu(self.aconv5(x))

        # Concatenate the outputs along channel dimension
        cat = torch.cat((out1, out2, out3, out4, out5), dim=1)

        # Apply batch normalization
        out = self.bn(cat)

        # Apply the prediction convolution
        pred = self.pred_conv(out)

        return pred
