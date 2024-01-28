# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
"""backbone:segformer; decoder:dahead(concat(c1+c2+c3+c4)); RGB+hpf   dual=Fasle 不使用hpf"""
"""加入cw_hpf,Co-Atttntion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_9_ import mit_b0_9, mit_b2_9
from .backbone import mit_b0, mit_b2
from .coordatt import CoordAtt


import numpy as np

def SRMLayer():
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = np.asarray(
        [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
    filters = np.repeat(filters,repeats=3, axis=0)
    filters = torch.from_numpy(filters.astype(np.float32))
    return filters

class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(DepthwiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,out_channels, kernel_size,stride,padding, dilation, groups=in_channels, bias=bias)
        filters = SRMLayer()
        self.depthwise.weight = nn.Parameter(filters)
        self.depthwise.weight.requires_grad =False
    def forward(self, x):
        out = self.depthwise(x)
        return out

class SegFormer(nn.Module):
    def __init__(self, num_classes = 2, phi = 'b5', pretrained = False, dual = False, **kwargs):
        super(SegFormer, self).__init__()
        self.dual = dual
        self.num_class = num_classes
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b2': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b2': mit_b2,
        }[phi](pretrained)
        self.backbone_9   = {
            'b0': mit_b0_9,  'b2': mit_b2_9,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256,  'b2': 768,
        }[phi]

        if self.dual:
            print("----------use constrain-------------")
            self.hpf_conv = DepthwiseConv2D(in_channels=3,out_channels=9, kernel_size=5, padding=2)
            self.head =CoordAtt(512+512,512+512,self.num_class,reduction=32)     ### 根据用的骨干网不同，需要修改！！！
        else:
            self.head =CoordAtt(512,self.num_class,reduction=32)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        c1, c2, c3, c4 = x
        if self.dual:
            channels = torch.split(inputs, split_size_or_sections=1, dim=1)
            new_inputs = torch.cat(channels * 3, dim=1)
            y = self.hpf_conv(new_inputs)

            hpf_features = self.backbone_9.forward(y)

            c4 = torch.cat([c4, hpf_features[3]], dim=1)
            c3 = torch.cat([c3, hpf_features[2]], dim=1)
            c2 = torch.cat([c2, hpf_features[1]], dim=1)
            c1 = torch.cat([c1, hpf_features[0]], dim=1)

        c4321 = torch.cat((c1, self.upsample2(c2),self.upsample4(c3), self.upsample8(c4)),1)
        x = self.head(c4321)
        x0 = F.interpolate(x[0], size=(H, W), mode='bilinear', align_corners=True)

        return x0
