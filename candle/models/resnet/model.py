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
                 resnet_blocks: List[Tuple[int, int, int]],
                 use_maxpool: bool = False):
        """
        Parameters
        ----------
        num_classes
            Num output classes
        in_channel
            Num channels in the input image.
        resnet_blocks
            List of (in_channels, out_channels, stride) tuples.
            
            e.g. for ResNet20m
                resnet_blocks = [
                    (16, 16, 1),
                    (16, 16, 1),
                    (16, 16, 1),

                    (16, 32, 2),
                    (32, 32, 1),
                    (32, 32, 1),

                    (32, 64, 2),
                    (64, 64, 1),
                    (64, 64, 1),
                ]
        use_maxpool
            If False, turns the MaxPool layer off.
        
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.resnet_blocks = resnet_blocks
        self.use_maxpool = use_maxpool
        
        self.conv = candle.Conv2d(in_channels,
                                  resnet_blocks[0][0],  # in-channels of the first resnet_block
                                  kernel_size=3, padding=1, stride=1)
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
        
        if self.use_maxpool:
            x = self.max_pool(x)
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            x = F.relu(x)
            
        x = x.mean(axis=(2, 3))  # Equivalent to global channel-wise avgpool
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
        
        # If channels are different or stride > 1, then we need to reshape the input using a 1x1 conv
        if in_channels != out_channels or stride > 1:
            self.conv_1by1 = candle.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv_1by1 = None
            
            
    def forward(self, x):
        x_resid = self.conv1(x)
        x_resid = self.batch_norm1(x_resid)
        x_resid = F.relu(x_resid)
        
        x_resid = self.conv2(x_resid)
        x_resid = self.batch_norm2(x_resid)
        
        if self.conv_1by1 is not None:
            x = self.conv_1by1(x)
            
        x_resid = x + x_resid
        
        return x_resid
    