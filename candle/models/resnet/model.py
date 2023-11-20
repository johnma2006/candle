"""Residual Network.

References:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015
    
"""

from typing import List, Tuple

import candle
import candle.functions as F
from candle.layers.module import Module


class ResNet(Module):
    
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 resnet_blocks: List[Tuple[int, int, int]]):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv = candle.Conv2d(in_channels, resnet_blocks[0][0], kernel_size=7, padding=1, stride=1)
        self.batch_norm = candle.BatchNorm(axis=(0, 2, 3))
        self.max_pool = candle.MaxPool2d(kernel_size=2)
        
        self.residual_blocks = candle.ParameterList([
            ResNetBlock(in_channels, out_channels, stride)
            for (in_channels, out_channels, stride) in resnet_blocks
        ])
        
        self.linear = candle.Linear(resnet_blocks[-1][1], num_classes)
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        
        x = self.max_pool(x)
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            x = F.relu(x)
            
        x = x.mean(axis=(2, 3))
        x = self.linear(x)
        
        return x
    
    
class ResNetBlock(Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = candle.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = candle.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.batch_norm1 = candle.BatchNorm(axis=(0, 2, 3))
        self.batch_norm2 = candle.BatchNorm(axis=(0, 2, 3))
        
        if in_channels != out_channels or stride > 1:
            self.res_conv = candle.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.res_conv = None
            
            
    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.batch_norm1(x_conv)
        x_conv = F.relu(x_conv)
        
        x_conv = self.conv2(x_conv)
        x_conv = self.batch_norm2(x_conv)
        
        if self.res_conv is not None:
            x = self.res_conv(x)
            
        x_conv = x + x_conv
        
        return x_conv
    