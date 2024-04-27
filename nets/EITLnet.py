# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone_9_ import mit_b0_9,  mit_b2_9
from .backbone import mit_b0,  mit_b2
from .coordatt import CoordAtt


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
    # print(filters.shape)
    filters = np.repeat(filters, repeats=3, axis=0)
    filters = torch.from_numpy(filters.astype(np.float32))
    # filters = torch.from_numpy(filters)
    # print(filters.shape)
    return filters


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        filters = SRMLayer()
        self.depthwise.weight = nn.Parameter(filters)
        self.depthwise.weight.requires_grad = False

    def forward(self, x):
        out = self.depthwise(x)
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=2, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = [64, 128, 320, 512]#b0
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = [128, 256, 640, 1024]  # b3
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=2, phi='b0', pretrained=False, dual=False):
        super(SegFormer, self).__init__()
        self.dual = dual
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0,  'b2': mit_b2,
        }[phi](pretrained)
        self.backbone_3 = {
            'b0': mit_b0_9, 'b2': mit_b2_9,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.channels = [64, 128, 320, 512]
        # self.channels = [32, 64, 160, 256]

        if self.dual:
            print("----------use constrain-------------")
            self.hpf_conv = DepthwiseConv2D(in_channels=3, out_channels=9, kernel_size=5, padding=2)
            self.caf4 = CoordAtt(self.channels[3], self.channels[3], reduction=32)
            self.caf3 = CoordAtt(self.channels[2], self.channels[2], reduction=32)
            self.caf2 = CoordAtt(self.channels[1], self.channels[1], reduction=32)
            self.caf1 = CoordAtt(self.channels[0], self.channels[0], reduction=32)
            self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        else:
            self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        c1, c2, c3, c4 = x
        if self.dual:
            channels = torch.split(inputs, split_size_or_sections=1, dim=1)
            new_inputs = torch.cat(channels * 3, dim=1)
            y = self.hpf_conv(new_inputs)
            residual_features = self.backbone_3.forward(y)  # hpf流
            c4 = torch.cat([c4, residual_features[3]], dim=1)
            c3 = torch.cat([c3, residual_features[2]], dim=1)
            c2 = torch.cat([c2, residual_features[1]], dim=1)
            c1 = torch.cat([c1, residual_features[0]], dim=1)
            c4 = self.caf4(c4)
            c3 = self.caf3(c3)
            c2 = self.caf2(c2)
            c1 = self.caf1(c1)
            x = [c1, c2, c3, c4]

        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
